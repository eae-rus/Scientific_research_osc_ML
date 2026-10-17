namespace OscFormats.Native
{
  using OscFormats.Model;
  using System;
  using System.Collections.Generic;
  using System.Globalization;
  using System.IO;
  using System.Linq;
  using System.Text;

  /// <summary>Блочный формат SGN_FILE регистраторов РЭС-3</summary>
  internal static class Res3Reader
  {
    private sealed class Channel
    {
      public int Position;
      public int Offset;
      public int Bit;
    }

    private static string Text(byte[] bytes, int offset, int length)
    {
      int end = Array.IndexOf(bytes, (byte)0, offset, length);
      return Encoding.GetEncoding(1251).GetString(bytes, offset, (end < 0 ? offset + length : end) - offset).Replace(',', '_');
    }

    private static double SixDigits(double value)
    {
      if (double.IsNaN(value) || double.IsInfinity(value)) { throw new InvalidDataException(); }
      return double.Parse(value.ToString("F6", CultureInfo.InvariantCulture), CultureInfo.InvariantCulture);
    }

    private static DateTime Timestamp(double seconds)
    {
      double whole = Math.Floor(seconds);
      return new DateTime(1970, 1, 1).AddSeconds(whole).AddMilliseconds((int)((seconds - whole) * 1000));
    }



    public static OscillogramData Read(string path, bool allowIncompleteFinalBlock = false)
    {
      byte[] bytes = File.ReadAllBytes(path);
      if (bytes.Length < 39 || Encoding.ASCII.GetString(bytes, 0, 9) != "SGN_FILE\0" || BitConverter.ToUInt16(bytes, 9) != 39) { throw new InvalidDataException(); }
      int version = BitConverter.ToUInt16(bytes, 15);
      if (version != 6 && version != 256) { throw new NotSupportedException(); }
      uint declaredLength = BitConverter.ToUInt32(bytes, 11);
      if (declaredLength != 0 && (declaredLength < bytes.Length || declaredLength > bytes.Length && !allowIncompleteFinalBlock)) { throw new InvalidDataException(); }
      double rate = BitConverter.ToDouble(bytes, 17);
      if (rate <= 0 || double.IsNaN(rate) || double.IsInfinity(rate)) { throw new InvalidDataException(); }
      uint seconds = BitConverter.ToUInt32(bytes, 25);
      double fraction = BitConverter.ToDouble(bytes, 29);
      if (fraction < 0 || fraction >= 1 || double.IsNaN(fraction)) { throw new InvalidDataException(); }
      var ports = new Dictionary<int, int>();
      var analogSources = new List<Channel>();
      var discreteSources = new List<Channel>();
      var blocks = new List<int>();
      var analogs = new List<AnalogChannel>();
      var discretes = new List<DiscreteChannel>();
      string station = null;
      int? triggerSample = null;
      for (int position = 39; position < bytes.Length;)
      {
        if (bytes.Length - position < 8) { throw new InvalidDataException(); }
        int kind = BitConverter.ToInt32(bytes, position);
        int length = BitConverter.ToInt32(bytes, position + 4);
        if (length < 8) { throw new InvalidDataException(); }
        if (length > bytes.Length - position)
        {
          // В режиме восстановления экспортируем только целые блоки, как SW_Converter
          if (allowIncompleteFinalBlock && kind == 4 && blocks.Count > 0) { break; }
          throw new InvalidDataException();
        }
        switch (kind)
        {
          case 0: break;
          case 1:
            if (length != 82 || station != null) { throw new InvalidDataException(); }
            station = Text(bytes, position + 18, 64);
            break;
          case 5:
            if (length != 12) { throw new NotSupportedException(); }
            ports.Add(BitConverter.ToUInt16(bytes, position + 8), BitConverter.ToUInt16(bytes, position + 10));
            break;
          case 2:
          case 3:
            if (length != (kind == 2 ? 160 : 71)) { throw new NotSupportedException(); }
            var channel = new Channel
            {
              Position = position,
              Offset = BitConverter.ToUInt16(bytes, position + 64),
              Bit = kind == 2 ? BitConverter.ToUInt16(bytes, position + 66) : bytes[position + 66]
            };
            string name = Text(bytes, position + 32, 32);
            if (kind == 2)
            {
              double scale = BitConverter.ToDouble(bytes, position + 80);
              double offset = BitConverter.ToDouble(bytes, position + 88) * scale;
              int unit = BitConverter.ToUInt16(bytes, position + 74);
              if (unit > 2) { throw new NotSupportedException(); }
              analogSources.Add(channel);
              analogs.Add(new AnalogChannel(analogs.Count + 1, name, "", "", unit == 1 ? "В" : unit == 2 ? "А" : "",
                SixDigits(scale), SixDigits(offset), 0, -32500, 32500, 1, 1, SignalType.Secondary));
            }
            else
            {
              discreteSources.Add(channel);
              discretes.Add(new DiscreteChannel(discretes.Count + 1, name, "", "", BitConverter.ToUInt16(bytes, position + 14) != 0));
            }
            break;
          case 4:
            if (length < 18) { throw new InvalidDataException(); }
            blocks.Add(position);
            break;
          case 6: break;
          case 8:
            if (length < 24) { throw new InvalidDataException(); }
            if (!triggerSample.HasValue) { triggerSample = BitConverter.ToInt32(bytes, position + 8); }
            break;
          default: throw new NotSupportedException();
        }
        position += length;
      }
      if (station == null || !triggerSample.HasValue || blocks.Count == 0 || ports.Count != 1) { throw new NotSupportedException(); }
      int size = ports.Values.Single();
      if (size == 0 || analogSources.Any(c => c.Offset + 2 > size || c.Bit != ushort.MaxValue) ||
        discreteSources.Any(c => c.Offset >= size || c.Bit == 0 || c.Bit > 128 || (c.Bit & (c.Bit - 1)) != 0)) { throw new NotSupportedException(); }
      int totalSampleCount = 0;
      foreach (int block in blocks)
      {
        if (BitConverter.ToUInt16(bytes, block + 12) != 1) { throw new NotSupportedException(); }
        int count = BitConverter.ToInt32(bytes, block + 14);
        if (count <= 0 || (long)size * count != BitConverter.ToInt32(bytes, block + 4) - 18) { throw new InvalidDataException(); }
        totalSampleCount += count;
      }
      float[][] analogColumns = new float[analogs.Count][];
      for (int c = 0; c < analogColumns.Length; c++) { analogColumns[c] = new float[totalSampleCount]; }
      bool[][] discreteColumns = new bool[discretes.Count][];
      for (int c = 0; c < discreteColumns.Length; c++) { discreteColumns[c] = new bool[totalSampleCount]; }
      int[] analogOffsets = analogSources.Select(c => c.Offset).ToArray();
      int[] discreteOffsets = discreteSources.Select(c => c.Offset).ToArray();
      byte[] discreteBits = discreteSources.Select(c => (byte)c.Bit).ToArray();
      int sampleIndex = 0;
      foreach (int block in blocks)
      {
        int count = BitConverter.ToInt32(bytes, block + 14);
        int blockEnd = block + 18 + count * size;
        for (int position = block + 18; position < blockEnd; position += size, sampleIndex++)
        {
          for (int c = 0; c < analogOffsets.Length; c++)
          {
            int p = position + analogOffsets[c];
            analogColumns[c][sampleIndex] = (short)(bytes[p] | (bytes[p + 1] << 8));
          }
          for (int c = 0; c < discreteOffsets.Length; c++)
          {
            discreteColumns[c][sampleIndex] = (bytes[position + discreteOffsets[c]] & discreteBits[c]) != 0;
          }
        }
      }
      var configuration = new ConfigurationHandler
      {
        StationName = station,
        DeviceId = "1",
        Version = ComtradeVersion.V1999,
        AnalogChannels = analogs,
        DiscreteChannels = discretes,
        SampleRates = new List<SampleRate> { new SampleRate(SixDigits(rate), totalSampleCount) },
        Frequency = 50,
        StartTime = Timestamp(seconds + fraction - triggerSample.Value / rate),
        TriggerTime = new DateTime(1970, 1, 1).AddSeconds(seconds).AddMilliseconds((int)(fraction * 1000)),
        DataFileType = DataFileType.ASCII,
        TimeMultiplicationFactor = 1
      };
      return NativeSampleBuilder.Create(configuration, analogColumns, discreteColumns);
    }
  }
}
