namespace OscFormats.Native
{
  using OscFormats.Model;
  using System;
  using System.Collections.Generic;
  using System.Globalization;
  using System.IO;
  using System.Linq;
  using System.Text;

  /// <summary>НЕВА 3.0: кадровая запись АЦП и поток событий БРКУ</summary>
  internal static class NevaReader
  {
    private static string Text(byte[] data, int offset, int length)
    {
      int end = Array.IndexOf(data, (byte)0, offset, length);
      return Encoding.GetEncoding(1251).GetString(data, offset, (end < 0 ? offset + length : end) - offset).Replace(',', '_').Trim();
    }

    private static double Decimal(float value)
    {
      if (float.IsNaN(value) || float.IsInfinity(value)) { throw new InvalidDataException(); }
      return double.Parse(((double)value).ToString("F6", CultureInfo.InvariantCulture), CultureInfo.InvariantCulture);
    }



    public static OscillogramData Read(string path, bool recoverIncomplete = false)
    {
      byte[] data = File.ReadAllBytes(path);
      if (data.Length < 1424 || Encoding.ASCII.GetString(data, 0, 80).TrimEnd((char)0) != "NEVA (version 3.0), Copyright (C) Saint-Petersburg, Energosoyouz" || BitConverter.ToUInt16(data, 81) != 3) { throw new InvalidDataException(); }
      int na = BitConverter.ToUInt16(data, 215), nd = BitConverter.ToUInt16(data, 217);
      if (na != 64 || nd == 0 || nd > 288) { throw new NotSupportedException(); }
      int header = checked(912 + na * 593 + nd * 108 + BitConverter.ToUInt16(data, 219) * 804 + BitConverter.ToUInt16(data, 221) * 274 + BitConverter.ToUInt16(data, 227) * 165);
      int start = checked(header + 512);
      if (start >= data.Length) { throw new InvalidDataException(); }
      int mode = BitConverter.ToUInt16(data, header + 384);
      if (mode != 0 && mode != 2) { throw new NotSupportedException(); }
      var trigger = new DateTime(BitConverter.ToUInt16(data, start - 9), data[start - 7], data[start - 6], data[start - 5], data[start - 4], data[start - 3], BitConverter.ToUInt16(data, start - 2));
      // В потоке событий дата заголовка записана в UTC, в кадровой записи — в местном времени
      if (mode == 2) { trigger = DateTime.SpecifyKind(trigger, DateTimeKind.Utc).ToLocalTime(); }
      var analogs = new List<AnalogChannel>();
      for (int i = 0; i < na; i++)
      {
        int p = 913 + i * 593;
        int phase = data[p + 38];
        string phaseName = phase >= 1 && phase <= 3 ? new[] { "A", "B", "C" }[phase - 1] : phase == 4 || phase == 5 ? "N" : "";
        analogs.Add(new AnalogChannel(i + 1, Text(data, p, 37), phaseName, "", Text(data, p + 39, 17),
          Decimal(BitConverter.ToSingle(data, p + 72)), 0, i % 16 * 20, -4096, 4095,
          Decimal(BitConverter.ToSingle(data, p + 80)), 1, SignalType.Primary));
      }
      var discretes = Enumerable.Range(0, nd).Select(i => new DiscreteChannel(i + 1, Text(data, 913 + na * 593 + i * 108, 37), "", "", false)).ToList();
      DateTime beginning;
      float[][] analogColumns;
      bool[][] discreteColumns;
      long[] timestampsUs = null;
      int totalCount = 0;
      int[] discreteByteOffsets = new int[nd];
      byte[] discreteBitMasks = new byte[nd];
      for (int j = 0; j < nd; j++)
      {
        discreteByteOffsets[j] = j / 8;
        discreteBitMasks[j] = (byte)(1 << (j % 8));
      }

      if (mode == 0)
      {
        int stride = na * 2 + (nd + 7) / 8;
        if ((data.Length - start) % stride != 0) { throw new InvalidDataException(); }
        // Сохраняем все полные кадры, включая пропускаемый штатным экспортом последний
        int count = (data.Length - start) / stride;
        totalCount = count;
        analogColumns = new float[na][];
        for (int c = 0; c < na; c++) { analogColumns[c] = new float[count]; }
        discreteColumns = new bool[nd][];
        for (int c = 0; c < nd; c++) { discreteColumns[c] = new bool[count]; }
        for (int i = 0; i < count; i++)
        {
          int p = start + i * stride;
          for (int j = 0; j < na; j++)
          {
            int raw = BitConverter.ToUInt16(data, p + j * 2);
            if (raw > 8191) { throw new InvalidDataException(); }
            analogColumns[j][i] = raw >= 4096 ? raw - 8192 : raw;
          }
          int dPos = p + na * 2;
          for (int j = 0; j < nd; j++)
          {
            discreteColumns[j][i] = (data[dPos + discreteByteOffsets[j]] & discreteBitMasks[j]) == 0;
          }
        }
        beginning = trigger.AddMilliseconds(-BitConverter.ToUInt16(data, 365));
      }
      else
      {
        uint current = 0, trip = 0, first = 0, seconds = 0;
        long nanoseconds = 0, origin = 0, frameTime = 0;
        bool precisionTime = false, active = false, tripFound = false;
        uint? firstPacket = null;
        bool firstPacketIsDigital = false, digitalKnown = false;
        byte[] frameBits = null;
        float[] a = new float[na];
        bool[] seen = new bool[na];
        int seenCount = 0;
        byte[] bits = new byte[36];

        int capacity = Math.Max(1024, (data.Length - start) / Math.Max(5, (na + 1) * 5));
        analogColumns = new float[na][];
        for (int c = 0; c < na; c++) { analogColumns[c] = new float[capacity]; }
        discreteColumns = new bool[nd][];
        for (int c = 0; c < nd; c++) { discreteColumns[c] = new bool[capacity]; }
        timestampsUs = new long[capacity];

        void EnsureCapacity(int size)
        {
          if (size <= analogColumns[0].Length) { return; }
          int newCap = Math.Max(size, analogColumns[0].Length * 2);
          for (int c = 0; c < na; c++) { Array.Resize(ref analogColumns[c], newCap); }
          for (int c = 0; c < nd; c++) { Array.Resize(ref discreteColumns[c], newCap); }
          Array.Resize(ref timestampsUs, newCap);
        }

        Action<bool> finish = final =>
        {
          if (!active) { return; }
          if (seenCount != na)
          {
            if (!recoverIncomplete || !final) { throw new InvalidDataException(); }
            return;
          }
          byte[] digital = frameBits ?? bits;
          EnsureCapacity(totalCount + 1);
          for (int c = 0; c < na; c++) { analogColumns[c][totalCount] = a[c]; }
          for (int j = 0; j < nd; j++)
          {
            discreteColumns[j][totalCount] = (digital[discreteByteOffsets[j]] & discreteBitMasks[j]) == 0;
          }
          timestampsUs[totalCount] = frameTime;
          totalCount++;
        };
        int position = start;
        for (; position + 5 <= data.Length; position += 5)
        {
          uint value = BitConverter.ToUInt32(data, position + 1);
          switch (data[position])
          {
            case 0: current = value; break;
            case 1:
              if (!firstPacket.HasValue) { firstPacket = current; }
              int channel = BitConverter.ToUInt16(data, position + 1);
              if (channel >= na) { throw new InvalidDataException(); }
              if (channel == 0)
              {
                finish(false);
                if (!active) { first = current; origin = seconds * 1000000000L + nanoseconds; }
                active = true;
                frameBits = digitalKnown ? (byte[])bits.Clone() : null;
                Array.Clear(seen, 0, seen.Length);
                seenCount = 0;
                frameTime = precisionTime ? (seconds * 1000000000L + nanoseconds - origin) / 1000 : unchecked(current - first);
                if (totalCount > 0 && frameTime <= timestampsUs[totalCount - 1]) { throw new InvalidDataException(); }
              }
              a[channel] = BitConverter.ToInt16(data, position + 3) / 2;
              if (!seen[channel]) { seen[channel] = true; seenCount++; }
              break;
            case 2:
              if (!firstPacket.HasValue) { firstPacket = current; firstPacketIsDigital = true; }
              digitalKnown = true;
              int x = data[position + 1], y = data[position + 2];
              if (x >= bits.Length || y >= bits.Length) { throw new InvalidDataException(); }
              bits[x] = data[position + 3]; bits[y] = data[position + 4];
              break;
            case 5: if (!tripFound) { trip = value; tripFound = true; } break;
            case 11: seconds = value; break;
            case 12: nanoseconds = value; break;
            case 13: precisionTime = value == 3; break;
            case 3: case 6: case 7: case 8: case 9: break;
            default: throw new NotSupportedException();
          }
        }
        if (position != data.Length && !recoverIncomplete) { throw new InvalidDataException(); }
        finish(true);
        if (!tripFound || !active) { throw new InvalidDataException(); }
        beginning = trigger.AddMilliseconds(Math.Round(unchecked((int)((firstPacketIsDigital ? firstPacket.Value : first) - trip)) / 1000.0));

        if (totalCount < analogColumns[0].Length)
        {
          for (int c = 0; c < na; c++) { Array.Resize(ref analogColumns[c], totalCount); }
          for (int c = 0; c < nd; c++) { Array.Resize(ref discreteColumns[c], totalCount); }
          Array.Resize(ref timestampsUs, totalCount);
        }
      }
      if (totalCount == 0) { throw new InvalidDataException(); }
      var configuration = new ConfigurationHandler
      {
        StationName = "NEVA",
        DeviceId = BitConverter.ToUInt16(data, 289).ToString("D4", CultureInfo.InvariantCulture),
        Version = ComtradeVersion.V1999,
        AnalogChannels = analogs,
        DiscreteChannels = discretes,
        SampleRates = new List<SampleRate> { new SampleRate(1000, totalCount) },
        Frequency = 50,
        StartTime = beginning,
        TriggerTime = trigger,
        DataFileType = DataFileType.BINARY,
        TimeMultiplicationFactor = 1
      };
      return timestampsUs != null
        ? NativeSampleBuilder.Create(configuration, analogColumns, discreteColumns, timestampsUs)
        : NativeSampleBuilder.Create(configuration, analogColumns, discreteColumns);
    }
  }
}
