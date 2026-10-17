namespace OscFormats.Native
{
  using OscFormats.Model;
  using System;
  using System.Collections.Generic;
  using System.Globalization;
  using System.IO;
  using System.Linq;
  using System.Text;

  /// <summary>Чтение WNDR без загрузки библиотек и запуска приложений ЭКРА/WinBres</summary>
  internal static class EkraDfrReader
  {
    private sealed class Segment
    {
      public string Path;
      public byte[] Bytes;
      public int Offset;
      public int Count;
      public int First;
    }

    private static Segment ReadSegment(string path)
    {
      byte[] data = File.ReadAllBytes(path);
      int position = 0;
      if (ReadLine(data, ref position) != "[WNDR]")
      {
        throw new InvalidDataException();
      }
      while (ReadLine(data, ref position) != "[Data]") { }
      if (data.Length - position < 256)
      {
        throw new InvalidDataException();
      }
      return new Segment
      {
        Path = path,
        Bytes = data,
        Offset = position,
        Count = BitConverter.ToUInt16(data, position + 12),
        First = BitConverter.ToUInt16(data, position + 95)
      };
    }

    private static string ReadLine(byte[] data, ref int position)
    {
      int start = position;
      while (position + 1 < data.Length)
      {
        if (data[position++] == 13 && data[position] == 10)
        {
          int length = position - start - 1;
          position++;
          return Encoding.GetEncoding(1251).GetString(data, start, length).Replace('\x05', ' ');
        }
      }
      throw new InvalidDataException();
    }

    private static bool SameRecording(Segment first, Segment candidate)
    {
      if (first.Offset != candidate.Offset) { return false; }
      for (int i = 0; i < first.Offset; i++)
      {
        if (first.Bytes[i] != candidate.Bytes[i]) { return false; }
      }
      for (int i = 0; i < 256; i++)
      {
        // Меняются номера частей, количество и первый отсчёт, контрольная сумма
        if (i == 10 || i == 11 || i == 12 || i == 13 || i == 95 || i == 96 || i == 255) { continue; }
        if (first.Bytes[first.Offset + i] != candidate.Bytes[candidate.Offset + i]) { return false; }
      }
      return true;
    }

    private static List<Segment> ReadSegments(string path)
    {
      var initial = ReadSegment(path);
      var candidates = new List<Segment> { initial };
      foreach (string candidatePath in Directory.EnumerateFiles(Path.GetDirectoryName(Path.GetFullPath(path))))
      {
        if (!string.Equals(Path.GetExtension(candidatePath), ".dfr", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(Path.GetFullPath(candidatePath), Path.GetFullPath(path), StringComparison.OrdinalIgnoreCase)) { continue; }
        Segment candidate;
        try { candidate = ReadSegment(candidatePath); }
        catch (InvalidDataException) { continue; }
        if (SameRecording(initial, candidate)) { candidates.Add(candidate); }
      }
      var segments = new List<Segment> { initial };
      foreach (int direction in new[] { -1, 1 })
      {
        Segment current = initial;
        while (true)
        {
          var next = candidates.Where(s => !segments.Contains(s) && s.Count > 0 &&
            (direction < 0 ? s.First + s.Count == current.First : current.First + current.Count == s.First)).ToList();
          if (next.Count == 0) { break; }
          if (next.Count != 1) { throw new InvalidDataException(); }
          segments.Add(next[0]);
          current = next[0];
        }
      }
      return segments.OrderBy(s => s.First).ToList();
    }

    private static string Clean(string value) => value.Trim().Trim('"').Trim();

    private static double? Number(string value)
    {
      return double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double number) &&
        !double.IsNaN(number) && !double.IsInfinity(number) ? number : (double?)null;
    }

    private static string Phase(string value, string name)
    {
      string[] phases = { "A", "B", "C" };
      string[] cyrillic = { "А", "В", "С" };
      value = value.Trim().ToUpperInvariant();
      for (int i = 0; i < phases.Length; i++)
      {
        if (value == phases[i] || value == cyrillic[i]) { return phases[i]; }
      }
      if (value.Length == 0)
      {
        name = name.ToUpperInvariant();
        for (int i = 0; i < phases.Length; i++)
        {
          if (name.Contains("I" + phases[i]) || name.Contains("U" + phases[i])) { return phases[i]; }
        }
      }
      return "";
    }



    public static OscillogramData Read(string path)
    {
      var segments = ReadSegments(path);
      Segment first = segments[0];
      if (first.First != 1 || first.Count == 0) { throw new InvalidDataException(); }
      byte[] data = first.Bytes;
      int position = 8;
      string terminal = ReadLine(data, ref position);
      int comma = terminal.LastIndexOf(',');
      string device = comma >= 0 && int.TryParse(terminal.Substring(comma + 1), out int deviceNumber)
        ? deviceNumber.ToString(CultureInfo.InvariantCulture) : "1";
      string station = Clean(comma >= 0 ? terminal.Substring(0, comma) : terminal).Replace(',', '.');
      string[] counts = ReadLine(data, ref position).Split(',');
      int analogCount = int.Parse(counts.Single(s => s.EndsWith("A", StringComparison.OrdinalIgnoreCase)).TrimEnd('A', 'a'), CultureInfo.InvariantCulture);
      int discreteCount = int.Parse(counts.Single(s => s.EndsWith("D", StringComparison.OrdinalIgnoreCase)).TrimEnd('D', 'd'), CultureInfo.InvariantCulture);
      var analogs = new List<AnalogChannel>();
      for (int i = 0; i < analogCount; i++)
      {
        string[] fields = ReadLine(data, ref position).Split(',');
        if (fields.Length < 6) { throw new InvalidDataException(); }
        string name = Clean(fields[1]);
        string primaryUnit = fields[2].Trim();
        bool kilo = primaryUnit.StartsWith("к", StringComparison.OrdinalIgnoreCase) || primaryUnit.StartsWith("k", StringComparison.OrdinalIgnoreCase);
        double? primary = Number(fields[4]);
        double? secondary = Number(fields[5]);
        double scale = primary.HasValue ? primary.Value * (kilo ? 1000 : 1) : secondary ?? 1;
        double ratio = primary.HasValue && secondary.HasValue ? primary.Value / secondary.Value * (kilo ? 1000 : 1) : 1;
        string unit = string.IsNullOrWhiteSpace(fields[3]) ? (kilo ? primaryUnit.Substring(1) : primaryUnit) : fields[3].Trim();
        if (unit == "В") { unit = "V"; }
        if (unit == "А") { unit = "A"; }
        analogs.Add(new AnalogChannel(i + 1, name, Phase(fields.Length > 7 ? fields[7] : "", name),
          fields.Length > 6 ? Clean(fields[6]) : "", unit, scale, 0, 0, short.MinValue, short.MaxValue,
          ratio, 1, primary.HasValue ? SignalType.Primary : SignalType.Secondary));
      }
      var discretes = new List<DiscreteChannel>();
      for (int i = 0; i < discreteCount; i++)
      {
        string[] fields = ReadLine(data, ref position).Split(',');
        if (fields.Length < 2) { throw new InvalidDataException(); }
        discretes.Add(new DiscreteChannel(i + 1, Clean(fields[1]), "", "", false));
      }
      // Экспорт WinBres привязывает каналы только к полным трёхфазным группам
      var connections = new HashSet<string>(analogs.GroupBy(c => c.CircuitComponent + "\0" + c.Unit)
        .Where(g => new[] { "A", "B", "C" }.All(phase => g.Any(c => c.Phase == phase)))
        .Select(g => g.Key));
      analogs = analogs.Select(c => connections.Contains(c.CircuitComponent + "\0" + c.Unit) ? c :
        new AnalogChannel(c.Number, c.Name, "", "", c.Unit, c.A, c.B, c.Skew, c.Min, c.Max,
          c.Primary, c.Secondary, c.SignalType)).ToList();
      int header = first.Offset;
      int rate = BitConverter.ToUInt16(data, header + 82);
      if (rate == 0) { throw new InvalidDataException(); }
      int milliseconds = Math.Max(0, BitConverter.ToUInt16(data, header + 85) - 10);
      int year = data[header + 91];
      var trigger = new DateTime((year > 50 ? 1900 : 2000) + year, data[header + 90], data[header + 89],
        data[header + 88], data[header + 87], milliseconds / 1000).AddMilliseconds(milliseconds % 1000);
      var start = trigger.AddMilliseconds(-(int)(BitConverter.ToInt16(data, header + 93) * 1000.0 / rate));
      int recordSize = checked(analogCount * 2 + ((discreteCount + 15) / 16) * 2);
      if (recordSize == 0) { throw new InvalidDataException(); }
      int totalSampleCount = 0;
      foreach (Segment segment in segments) { totalSampleCount += segment.Count; }
      float[][] analogColumns = new float[analogCount][];
      for (int c = 0; c < analogCount; c++) { analogColumns[c] = new float[totalSampleCount]; }
      bool[][] discreteColumns = new bool[discreteCount][];
      for (int c = 0; c < discreteCount; c++) { discreteColumns[c] = new bool[totalSampleCount]; }
      int[] discreteByteOffsets = new int[discreteCount];
      byte[] discreteBitMasks = new byte[discreteCount];
      for (int j = 0; j < discreteCount; j++)
      {
        discreteByteOffsets[j] = j / 8;
        discreteBitMasks[j] = (byte)(1 << (j % 8));
      }
      int sampleIndex = 0;
      foreach (Segment segment in segments)
      {
        int segmentHeader = 8;
        ReadLine(segment.Bytes, ref segmentHeader);
        if (ReadLine(segment.Bytes, ref segmentHeader) != string.Join(",", counts) ||
          BitConverter.ToUInt16(segment.Bytes, segment.Offset + 82) != rate)
        {
          throw new InvalidDataException();
        }
        int offset = segment.Offset + 256;
        if ((long)offset + (long)recordSize * segment.Count > segment.Bytes.Length) { throw new InvalidDataException(); }
        for (int i = 0; i < segment.Count; i++, offset += recordSize, sampleIndex++)
        {
          int aPos = offset;
          for (int j = 0; j < analogCount; j++, aPos += 2)
          {
            analogColumns[j][sampleIndex] = (short)(segment.Bytes[aPos] | (segment.Bytes[aPos + 1] << 8));
          }
          int dPos = offset + 2 * analogCount;
          for (int j = 0; j < discreteCount; j++)
          {
            discreteColumns[j][sampleIndex] = (segment.Bytes[dPos + discreteByteOffsets[j]] & discreteBitMasks[j]) != 0;
          }
        }
      }
      var configuration = new ConfigurationHandler
      {
        StationName = station,
        DeviceId = device,
        Version = ComtradeVersion.V1999,
        AnalogChannels = analogs,
        DiscreteChannels = discretes,
        SampleRates = new List<SampleRate> { new SampleRate(rate, totalSampleCount) },
        Frequency = 50,
        StartTime = start,
        TriggerTime = trigger,
        DataFileType = DataFileType.BINARY,
        TimeMultiplicationFactor = 1
      };
      return NativeSampleBuilder.Create(configuration, analogColumns, discreteColumns);
    }
  }
}
