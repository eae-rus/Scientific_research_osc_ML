namespace OscFormats.Native
{
  using OscFormats.Model;
  using System;
  using System.Collections.Generic;
  using System.Globalization;
  using System.IO;
  using System.IO.Compression;
  using System.Linq;
  using System.Text;
  using System.Text.RegularExpressions;

  /// <summary>Бреслер 0107, версия записи 2, встроенные именованные уставки</summary>
  internal static class BreslerReader
  {
    private sealed class SettingsReader
    {
      private readonly BinaryReader mReader;
      private readonly Dictionary<int, List<Tuple<string, string>>> mSchemas = new Dictionary<int, List<Tuple<string, string>>>();
      public readonly Dictionary<int, List<Dictionary<string, object>>> Groups = new Dictionary<int, List<Dictionary<string, object>>>();

      public SettingsReader(byte[] bytes)
      {
        using (var stream = new MemoryStream(bytes))
        using (mReader = new BinaryReader(stream, Encoding.GetEncoding(1251)))
        {
          if (mReader.ReadInt32() != 369431374) { throw new NotSupportedException(); }
          Skip(20);
          while (Peek(0) == 0x55555555) { ReadGroup(0); }
        }
      }

      private int Peek(int offset)
      {
        long position = mReader.BaseStream.Position;
        if (mReader.BaseStream.Length - position < offset + 4) { return -1; }
        mReader.BaseStream.Position += offset;
        int value = mReader.ReadInt32();
        mReader.BaseStream.Position = position;
        return value;
      }

      private void Skip(int count)
      {
        if (count < 0 || count > mReader.BaseStream.Length - mReader.BaseStream.Position) { throw new InvalidDataException(); }
        mReader.BaseStream.Position += count;
      }

      private int Count()
      {
        int count = mReader.ReadInt32();
        if (count < 0 || count > mReader.BaseStream.Length - mReader.BaseStream.Position) { throw new InvalidDataException(); }
        return count;
      }

      private string Text()
      {
        return Encoding.GetEncoding(1251).GetString(mReader.ReadBytes(Count()));
      }

      private List<Tuple<string, string>> Fields()
      {
        int count = Count();
        var fields = new List<Tuple<string, string>>();
        for (int i = 0; i < count; i++) { fields.Add(Tuple.Create(Text(), Text())); }
        return fields;
      }

      private object Value(string type)
      {
        Text();
        mReader.ReadInt32();
        switch (type)
        {
          case "TStringParam": return Text();
          case "TIntParam":
          case "TUIntParam":
            int integer = mReader.ReadInt32();
            Skip(8);
            return integer;
          case "TRealParam":
            float real = mReader.ReadSingle();
            Skip(8);
            if (float.IsNaN(real) || float.IsInfinity(real)) { throw new InvalidDataException(); }
            return real;
          default: throw new NotSupportedException(type);
        }
      }

      private void ReadGroup(int expectedLevel)
      {
        if (expectedLevel > 32 || mReader.ReadInt32() != 0x55555555 || mReader.ReadInt32() != expectedLevel) { throw new InvalidDataException(); }
        int kind = mReader.ReadInt32();
        int components = mReader.ReadInt32();
        int subtypes = mReader.ReadInt32();
        int count = Count();
        if (components < 0 || components > 1 || subtypes < 0 || subtypes > 1) { throw new InvalidDataException(); }
        // Физические дискреты, выходные реле и логика используют общую схему полей
        int schemaKey = kind == 5 || kind == 10 ? 4 : kind;
        if (!Groups.TryGetValue(kind, out var group)) { Groups[kind] = group = new List<Dictionary<string, object>>(); }
        for (int i = 0; i < count; i++)
        {
          if (subtypes != 0) { mReader.ReadInt32(); }
          if (!mSchemas.TryGetValue(schemaKey, out var fields))
          {
            Text();
            mSchemas[schemaKey] = fields = Fields();
          }
          var additional = Fields();
          var values = new Dictionary<string, object>();
          foreach (var field in fields.Concat(additional)) { values.Add(field.Item2, Value(field.Item1)); }
          group.Add(values);
          if (components != 0)
          {
            int inputs = Count();
            for (int j = 0; j < inputs; j++) { Skip(checked(Count() + 8)); }
          }
          while (Peek(0) == 0x55555555 && Peek(4) > expectedLevel) { ReadGroup(expectedLevel + 1); }
        }
      }
    }

    private static byte[] Inflate(byte[] bytes)
    {
      if (bytes.Length < 6 || (bytes[0] & 15) != 8 || (bytes[0] * 256 + bytes[1]) % 31 != 0 || (bytes[1] & 32) != 0) { throw new InvalidDataException(); }
      using (var input = new MemoryStream(bytes, 2, bytes.Length - 6))
      using (var deflate = new DeflateStream(input, CompressionMode.Decompress))
      using (var output = new MemoryStream())
      {
        byte[] buffer = new byte[8192];
        int length;
        uint a = 1, b = 0;
        while ((length = deflate.Read(buffer, 0, buffer.Length)) != 0)
        {
          if (output.Length + length > 32 * 1024 * 1024) { throw new InvalidDataException(); }
          for (int i = 0; i < length; i++) { a = (a + buffer[i]) % 65521; b = (b + a) % 65521; }
          output.Write(buffer, 0, length);
        }
        int end = bytes.Length - 4;
        uint checksum = (uint)bytes[end] << 24 | (uint)bytes[end + 1] << 16 | (uint)bytes[end + 2] << 8 | bytes[end + 3];
        if (checksum != (b << 16 | a)) { throw new InvalidDataException(); }
        return output.ToArray();
      }
    }

    private static string Text(Dictionary<string, object> values, string key) => (string)values[key];
    private static double Real(Dictionary<string, object> values, string key) => System.Convert.ToDouble(values[key], CultureInfo.InvariantCulture);
    private static string Clean(string value) => value.Replace(',', '.').Replace('\r', ' ').Replace('\n', ' ');

    private static int ChannelIndex(string value)
    {
      var match = Regex.Match(value, @"\[(\d+)\]");
      return match.Success && int.TryParse(match.Groups[1].Value, out int number) ? number - 1 : -1;
    }

    private static void Align(BinaryReader reader) => reader.BaseStream.Position = (reader.BaseStream.Position + 3) / 4 * 4;

    private static int[] Registered(BinaryReader reader, int count)
    {
      int size = (count + 7) / 8;
      byte[] flags = reader.ReadBytes(size);
      if (flags.Length != size || reader.ReadBytes(size).Length != size) { throw new InvalidDataException(); }
      return Enumerable.Range(0, count).Where(i => (flags[i / 8] & (1 << (i % 8))) != 0).ToArray();
    }

    /// <summary>Экспорт физических входов, как в пакетном экспорте WinBres</summary>


    public static OscillogramData Read(string path)
    {
      byte[] bytes = File.ReadAllBytes(path);
      if (bytes.Length < 192 || bytes[0] != 2 || bytes[1] != 12 || bytes[2] != 7 || bytes[4] == 11) { throw new NotSupportedException(); }
      using (var reader = new BinaryReader(new MemoryStream(bytes)))
      {
        reader.BaseStream.Position = 8;
        int count = reader.ReadInt32();
        int dataOffset = reader.ReadInt32();
        if (count <= 0 || dataOffset < 192 || dataOffset > bytes.Length) { throw new InvalidDataException(); }
        var start = new DateTime(2000 + bytes[149], bytes[148], bytes[147], bytes[146], bytes[145], bytes[144], BitConverter.ToUInt16(bytes, 150) % 1000);
        reader.BaseStream.Position = 152;
        uint adc = reader.ReadUInt32();
        int divisor = reader.ReadInt32();
        reader.BaseStream.Position = 174;
        int rate = reader.ReadUInt16();
        if (adc == 0 || divisor <= 0 || rate == 0) { throw new InvalidDataException(); }
        reader.BaseStream.Position = 182;
        int na = reader.ReadUInt16(), nd = reader.ReadUInt16(), nr = reader.ReadUInt16(), nl = reader.ReadUInt16();
        reader.ReadInt16();
        if (na == 0 || (long)na * 13 + 192 > dataOffset) { throw new InvalidDataException(); }
        double[] scale = new double[na];
        double[] ratio = new double[na];
        for (int i = 0; i < na; i++) { scale[i] = reader.ReadUInt32() / (double)divisor / adc * Math.Sqrt(2); }
        reader.BaseStream.Position += 4 * na;
        for (int i = 0; i < na; i++)
        {
          ratio[i] = reader.ReadSingle();
          if (ratio[i] <= 0 || double.IsNaN(ratio[i]) || double.IsInfinity(ratio[i])) { throw new InvalidDataException(); }
        }
        byte[] types = reader.ReadBytes(na);
        reader.BaseStream.Position += 4 - na % 4;
        int trips = reader.ReadUInt16();
        reader.BaseStream.Position += trips * 2;
        Align(reader);
        int[] discreteIndices = Registered(reader, nd);
        int[] logicIndices = Registered(reader, nl);
        int[] relayIndices = Registered(reader, nr);
        Align(reader);
        int settingsLength = reader.ReadInt32();
        if (settingsLength <= 0 || reader.BaseStream.Position + settingsLength > dataOffset) { throw new InvalidDataException(); }
        var settings = new SettingsReader(Inflate(reader.ReadBytes(settingsLength))).Groups;
        if (settings[1].Count != na || settings[4].Count != nd) { throw new InvalidDataException(); }
        int digitalBytes = (discreteIndices.Length + 7) / 8;
        int internalBytes = (relayIndices.Length + 7) / 8 + (logicIndices.Length + 7) / 8;
        int recordSize = checked(na * 2 + digitalBytes + internalBytes);
        if ((long)dataOffset + (long)recordSize * count > bytes.Length) { throw new InvalidDataException(); }
        string[] phases = new string[na];
        string[] connections = new string[na];
        if (settings.TryGetValue(2, out var bays))
        {
          foreach (var bay in bays)
          {
            int[] indices = new[] { "PhaseA", "PhaseB", "PhaseC", "ZeroSeqChannel" }.Select(key => ChannelIndex(Text(bay, key))).ToArray();
            if (indices.Take(3).Any(i => i < 0 || i >= na)) { continue; }
            for (int i = 0; i < indices.Length; i++)
            {
              if (indices[i] < 0 || indices[i] >= na) { continue; }
              phases[indices[i]] = new[] { "A", "B", "C", "N" }[i];
              connections[indices[i]] = Clean(Text(bay, "Name"));
            }
          }
        }
        var analogs = new List<AnalogChannel>();
        for (int i = 0; i < na; i++)
        {
          var channel = settings[1][i];
          if (types[i] > 2) { throw new NotSupportedException(); }
          string secondary = Text(channel, "SecondaryUnit"), primary = Text(channel, "PrimaryUnit");
          string unit = secondary.Length > 0 && primary.Contains(secondary) ? secondary : primary;
          double offset = Real(channel, "Offset") * ratio[i];
          if (types[i] == 2)
          {
            scale[i] /= Math.Sqrt(2);
            float sensorScale = (float)((float)(Real(channel, "SensorPhysMaxValue") - Real(channel, "SensorPhysMinValue")) /
              (float)(Real(channel, "SensorMaxValue") - Real(channel, "SensorMinValue")));
            offset += (float)Real(channel, "SensorPhysMaxValue") - sensorScale * (float)Real(channel, "SensorMaxValue");
          }
          analogs.Add(new AnalogChannel(i + 1, Clean(Text(channel, "Name")), phases[i] ?? "", connections[i] ?? "", unit,
            scale[i] * ratio[i], offset, 0, short.MinValue, short.MaxValue, ratio[i], 1, SignalType.Primary));
        }
        var discretes = discreteIndices.Select((index, number) => new DiscreteChannel(number + 1,
          Text(settings[4][index], "Name"), "", "", false)).ToList();
        float[][] analogColumns = new float[na][];
        for (int c = 0; c < na; c++) { analogColumns[c] = new float[count]; }
        bool[][] discreteColumns = new bool[discreteIndices.Length][];
        for (int c = 0; c < discreteColumns.Length; c++) { discreteColumns[c] = new bool[count]; }
        int[] discreteByteOffsets = new int[discreteIndices.Length];
        byte[] discreteBitMasks = new byte[discreteIndices.Length];
        for (int j = 0; j < discreteIndices.Length; j++)
        {
          discreteByteOffsets[j] = j / 8;
          discreteBitMasks[j] = (byte)(1 << (j % 8));
        }
        int pos = dataOffset;
        for (int i = 0; i < count; i++, pos += recordSize)
        {
          int aPos = pos;
          for (int j = 0; j < na; j++, aPos += 2)
          {
            analogColumns[j][i] = (short)(bytes[aPos] | (bytes[aPos + 1] << 8));
          }
          int dPos = pos + na * 2;
          for (int j = 0; j < discreteIndices.Length; j++)
          {
            discreteColumns[j][i] = (bytes[dPos + discreteByteOffsets[j]] & discreteBitMasks[j]) != 0;
          }
        }
        var configuration = new ConfigurationHandler
        {
          StationName = Clean(Text(settings[0][0], "FSubstationName")),
          DeviceId = Clean(Text(settings[0][0], "FTerminalName")),
          Version = ComtradeVersion.V1999,
          AnalogChannels = analogs,
          DiscreteChannels = discretes,
          SampleRates = new List<SampleRate> { new SampleRate(rate, count) },
          Frequency = 50,
          StartTime = start,
          TriggerTime = start,
          DataFileType = DataFileType.BINARY,
          TimeMultiplicationFactor = 1
        };
        return NativeSampleBuilder.Create(configuration, analogColumns, discreteColumns);
      }
    }
  }
}
