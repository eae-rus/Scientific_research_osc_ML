namespace OscFormats.Native
{
  using OscFormats.Model;
  using System;
  using System.Collections.Generic;
  using System.Globalization;
  using System.IO;
  using System.Linq;
  using System.Text;
  using System.Text.RegularExpressions;

  /// <summary>Чтение записей DR терминалов ЭКРА с отдельным WConfig</summary>
  internal static class EkraNdrReader
  {
    private static void Require(bool condition)
    {
      if (!condition) { throw new InvalidDataException(); }
    }

    private static byte[] Unpack(byte[] source)
    {
      if (source.Length < 3 || source[0] != 'L' || source[1] != 'Z' || source[2] != 'M') { return source; }
      Require(source.Length >= 7);
      uint length = BitConverter.ToUInt32(source, 3);
      // Один поток DR содержит не более 2328 кадров и 64 байт заголовка
      Require(length >= 64 && length <= 64 + 2328 * 28);
      byte[] result = new byte[length];
      int position = 7, written = 0;
      while (position < source.Length)
      {
        Require(position + 2 <= source.Length);
        int size = BitConverter.ToUInt16(source, position);
        position += 2;
        int end = position + size;
        Require(size > 0 && end <= source.Length);
        byte[] block = new byte[8192];
        int count = 0;
        while (position < end)
        {
          int code = source[position++], literals = 0, match = 0, distance = 0;
          int type = code & 3;
          if (type == 0)
          {
            literals = code >> 2;
            if (literals == 0)
            {
              Require(position < end);
              int repeated = source[position++];
              if (repeated > 0)
              {
                Require(position < end && count + repeated <= block.Length);
                byte value = source[position++];
                for (int i = 0; i < repeated; i++) { block[count++] = value; }
              }
            }
          }
          else
          {
            Require(position + (type == 3 ? 2 : 1) <= end);
            code |= source[position++] << 8;
            if (type == 1)
            {
              literals = (code >> 2) & 3;
              match = ((code >> 4) & 7) + 3;
              distance = (code >> 7) & 511;
            }
            else if (type == 2)
            {
              match = ((code >> 2) & 3) + 3;
              distance = (code >> 4) & 4095;
            }
            else
            {
              code |= source[position++] << 16;
              literals = (code >> 2) & 15;
              match = (code >> 6) & 63;
              distance = (code >> 12) & 4095;
            }
          }
          Require(position + literals <= end && count + literals + match <= block.Length);
          Array.Copy(source, position, block, count, literals);
          count += literals;
          position += literals;
          if (match > 0)
          {
            Require(match <= distance && distance <= count);
            Array.Copy(block, count - distance, block, count, match);
            count += match;
          }
        }
        Require(count > 0 && written + count <= result.Length);
        Array.Copy(block, 0, result, written, count);
        written += count;
      }
      Require(written == result.Length);
      return result;
    }

    private static byte[] ReadPart(string path)
    {
      byte[] data = Unpack(File.ReadAllBytes(path));
      Require(data.Length >= 92 && (data.Length - 64) % 28 == 0 && data[61] < 4);
      int checksum = 0;
      for (int i = 0; i < 63; i++) { checksum += data[i]; }
      Require(((checksum ^ 255) & 255) == data[63]);
      int total = BitConverter.ToUInt16(data, 59) * 6;
      int count = Math.Min(2328, total - data[61] * 2328);
      Require(count > 0 && data.Length == 64 + count * 28);
      return data;
    }

    private static List<byte[]> ReadParts(string path)
    {
      byte[] initial = ReadPart(path);
      var parts = new List<byte[]> { initial };
      string stem = Path.GetFileNameWithoutExtension(path);
      Match match = Regex.Match(stem, @"^DR\d+F([0-3])$", RegexOptions.IgnoreCase);
      Require(match.Success && int.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture) == initial[61]);
      for (int part = 0; part < 4; part++)
      {
        if (part == initial[61]) { continue; }
        string sibling = Path.Combine(Path.GetDirectoryName(path), stem.Substring(0, stem.Length - 1) + part + Path.GetExtension(path));
        if (!File.Exists(sibling)) { continue; }
        byte[] next = ReadPart(sibling);
        Require(next[61] == part && next.Take(7).SequenceEqual(initial.Take(7)) &&
          next.Skip(46).Take(3).SequenceEqual(initial.Skip(46).Take(3)) &&
          next.Skip(54).Take(7).SequenceEqual(initial.Skip(54).Take(7)));
        parts.Add(next);
      }
      parts = parts.OrderBy(p => p[61]).ToList();
      for (int i = 1; i < parts.Count; i++)
      {
        Require(parts[i][61] == parts[i - 1][61] + 1 && parts[i - 1].Length == 64 + 2328 * 28);
      }
      return parts;
    }

    private static double Number(string value)
    {
      double result = double.Parse(value.Trim(), CultureInfo.InvariantCulture);
      Require(!double.IsNaN(result) && !double.IsInfinity(result));
      return result;
    }



    public static OscillogramData Read(string path, string configPath = null)
    {
      List<byte[]> parts = ReadParts(path);
      byte[] data = parts[0];
      string extension = Path.GetExtension(path);
      configPath = configPath ?? Directory.EnumerateFiles(Path.GetDirectoryName(Path.GetFullPath(path)))
        .SingleOrDefault(p => string.Equals(Path.GetFileName(p), "WConfig" + extension, StringComparison.OrdinalIgnoreCase));
      if (configPath == null) { throw new FileNotFoundException(null, "WConfig" + extension); }
      string[] lines = File.ReadAllLines(configPath, Encoding.GetEncoding(1251));
      Require(lines.Length >= 53 && lines[0] == "[WNDR]");
      string[] terminal = lines[1].Split(',');
      Require(terminal.Length == 2 && lines[2] == "48,16A,32D");
      Require(lines[51].Trim() == "2048" && lines[52].Trim() == "12");
      int device = data[47] * 100 + data[48];
      Require(int.Parse(terminal[1], CultureInfo.InvariantCulture) == device);
      var analogs = new List<AnalogChannel>();
      for (int i = 0; i < 16; i++)
      {
        string[] fields = lines[3 + i].Split(',');
        Require(fields.Length >= 10 && int.Parse(fields[0], CultureInfo.InvariantCulture) == i + 1);
        int type = int.Parse(fields[9], CultureInfo.InvariantCulture);
        double scale = Number(fields[4]), secondaryScale = Number(fields[5]);
        Require(secondaryScale != 0);
        string unit;
        double secondary, multiplier;
        switch (type)
        {
          case 13:
          case 15: unit = "kV"; secondary = 0.1; multiplier = 100; break;
          case 17: unit = "mA/V"; secondary = 100; multiplier = 100; break;
          case 22: unit = "A"; secondary = 1; multiplier = 1; break;
          default: throw new NotSupportedException();
        }
        double primary = Math.Round(scale / secondaryScale * multiplier, 1);
        analogs.Add(new AnalogChannel(i + 1, fields[1].Trim(), fields[7].Trim(), "", unit,
          scale, 0, 0, short.MinValue, short.MaxValue, primary, secondary, SignalType.Primary));
      }
      var digitalIndices = new List<int>();
      var discretes = new List<DiscreteChannel>();
      for (int i = 0; i < 32; i++)
      {
        string[] fields = lines[19 + i].Split(',');
        Require(fields.Length == 2 && int.Parse(fields[0], CultureInfo.InvariantCulture) == i + 1);
        string name = fields[1].Trim();
        if (name.Length == 0) { continue; }
        digitalIndices.Add(i);
        discretes.Add(new DiscreteChannel(discretes.Count + 1, (i + 1).ToString(CultureInfo.InvariantCulture) + " " + name, "", "", false));
      }
      double frequency = 50;
      if (((data[46] == 1 || data[46] == 3) && (data[23] & 128) != 0) || (data[46] == 2 && (data[24] & 2) != 0))
      {
        int timer = data[56] < 11 ? BitConverter.ToUInt16(data, 54) : 0xf900 + data[54];
        frequency = 1000.2 / ((65535 - timer + 13) * 0.012);
      }
      double rate = frequency * 12;
      Require(rate > 0);
      int milliseconds = data[56] < 8 ? data[1] * 1000 : BitConverter.ToUInt16(data, 0);
      var trigger = new DateTime((data[6] < 80 ? 2000 : 1900) + data[6], data[5], data[4], data[3], data[2], 0).AddMilliseconds(milliseconds);
      int pretrigger = data[21] == 0 ? 0 : data[21] * 12 - 2;
      int first = data[61] * 2328;
      var start = trigger.AddTicks((long)Math.Round((first - pretrigger) * 10000000.0 / rate));
      int totalSampleCount = 0;
      foreach (byte[] part in parts) { totalSampleCount += (part.Length - 64) / 28; }
      float[][] analogColumns = new float[16][];
      for (int c = 0; c < 16; c++) { analogColumns[c] = new float[totalSampleCount]; }
      bool[][] discreteColumns = new bool[digitalIndices.Count][];
      for (int c = 0; c < discreteColumns.Length; c++) { discreteColumns[c] = new bool[totalSampleCount]; }
      int[] discreteOffsets = new int[digitalIndices.Count];
      byte[] discreteMasks = new byte[digitalIndices.Count];
      for (int i = 0; i < digitalIndices.Count; i++)
      {
        int idx = digitalIndices[i];
        discreteOffsets[i] = 24 + idx / 8;
        discreteMasks[i] = (byte)(1 << (idx % 8));
      }
      int sampleIndex = 0;
      foreach (byte[] part in parts)
      {
        data = part;
        for (int offset = 64; offset < data.Length; offset += 28, sampleIndex++)
        {
          for (int pair = 0; pair < 8; pair++)
          {
            int position = offset + pair * 3;
            int x = data[position] | ((data[position + 2] & 240) << 4);
            int y = data[position + 1] | ((data[position + 2] & 15) << 8);
            analogColumns[2 * pair][sampleIndex] = x >= 2048 ? x - 4096 : x;
            analogColumns[2 * pair + 1][sampleIndex] = y >= 2048 ? y - 4096 : y;
          }
          for (int i = 0; i < discreteColumns.Length; i++)
          {
            discreteColumns[i][sampleIndex] = (data[offset + discreteOffsets[i]] & discreteMasks[i]) != 0;
          }
        }
      }
      var configuration = new ConfigurationHandler
      {
        StationName = terminal[0],
        DeviceId = terminal[1],
        Version = ComtradeVersion.V1999,
        AnalogChannels = analogs,
        DiscreteChannels = discretes,
        SampleRates = new List<SampleRate> { new SampleRate(Math.Round(rate, 6), totalSampleCount) },
        Frequency = frequency,
        StartTime = start,
        TriggerTime = trigger,
        DataFileType = DataFileType.BINARY,
        TimeMultiplicationFactor = 1
      };
      return NativeSampleBuilder.Create(configuration, analogColumns, discreteColumns);
    }
  }
}
