namespace OscFormats.Native
{
  using OscFormats.Model;
  using System;
  using System.Collections.Generic;
  using System.Globalization;
  using System.IO;
  using System.Linq;
  using System.Runtime.CompilerServices;
  using System.Text;
  using System.Threading.Tasks;

  /// <summary>Чёрный ящик: база KEBD и потоки версии 6</summary>
  internal static class BlackBoxReader
  {
    private static readonly CultureInfo mInvariant = CultureInfo.InvariantCulture;
    private static readonly Encoding mCp1251 = Encoding.GetEncoding(1251);
    [ThreadStatic]
    private static byte[] mTextBuffer;

    private abstract class Bits
    {
      public abstract int Bit();
      public abstract uint ReadBits(int count);
    }

    private sealed class Input : Bits, IDisposable
    {
      public readonly BinaryReader Reader;
      private int mCachedByte, mUsedBitCount = 8;
      public Input(byte[] bytes) { Reader = new BinaryReader(new MemoryStream(bytes), mCp1251); }
      public long Remaining => Reader.BaseStream.Length - Reader.BaseStream.Position;
      public void Dispose() => Reader.Dispose();
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      public override int Bit()
      {
        // Байтовые поля не выравнивают поток битов: непрочитанные биты остаются в кэше
        if (mUsedBitCount == 8) { mCachedByte = Reader.ReadByte(); mUsedBitCount = 0; }
        return mCachedByte >> mUsedBitCount++ & 1;
      }
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      public override uint ReadBits(int count)
      {
        if (count == 0) { return 0; }
        if (count < 0 || count > 32) { throw new InvalidDataException(); }
        int avail = 8 - mUsedBitCount;
        if (avail >= count)
        {
          uint value = (uint)((mCachedByte >> mUsedBitCount) & ((1 << count) - 1));
          mUsedBitCount += count;
          return value;
        }
        uint result = (uint)(mCachedByte >> mUsedBitCount);
        int position = avail;
        mUsedBitCount = 8;
        while (count - position >= 8)
        {
          result |= (uint)Reader.ReadByte() << position;
          position += 8;
        }
        int remaining = count - position;
        if (remaining > 0)
        {
          mCachedByte = Reader.ReadByte();
          result |= (uint)(mCachedByte & ((1 << remaining) - 1)) << position;
          mUsedBitCount = remaining;
        }
        return result;
      }
      public int Count(int maximum = 1000000)
      {
        int value = Reader.ReadInt32();
        if (value < 0 || value > maximum || value > Remaining) { throw new InvalidDataException(); }
        return value;
      }
      public string Text()
      {
        if (mTextBuffer == null) { mTextBuffer = new byte[1024]; }
        int len = 0;
        for (int i = 0; i < 65536; i++)
        {
          byte value = Reader.ReadByte();
          if (value == 0) { return mCp1251.GetString(mTextBuffer, 0, len); }
          if (len == mTextBuffer.Length) { Array.Resize(ref mTextBuffer, mTextBuffer.Length * 2); }
          mTextBuffer[len++] = value;
        }
        throw new InvalidDataException();
      }
      public byte[] Blob()
      {
        uint size = Reader.ReadUInt16();
        if ((size & 1) != 0) { size |= (uint)Reader.ReadUInt16() << 16; }
        size >>= 1;
        if (size > Remaining || size > 256 * 1024 * 1024) { throw new InvalidDataException(); }
        return Reader.ReadBytes((int)size);
      }
      public object Value(int type, bool schema = false)
      {
        switch (type)
        {
          case 1: return Reader.ReadInt32();
          case 3: return (int)Reader.ReadInt16();
          case 5: return (int)Reader.ReadByte();
          case 7: return Finite(Reader.ReadSingle());
          case 9: return Text();
          case 10: return schema ? null : Blob();
          default: throw new NotSupportedException();
        }
      }
    }

    private sealed class RunBits : Bits
    {
      private readonly Bits mSource;
      private readonly int mRunThreshold, mInitialBitWidth;
      private int mRemainingBitCount, mCurrentBit, mLastBit, mSameBitCount;
      private bool mFlipNextBit;
      private RunBits(Bits source)
      {
        this.mSource = source;
        mRunThreshold = (int)source.ReadBits(4);
        mInitialBitWidth = (int)source.ReadBits(4);
      }
      public override uint ReadBits(int count)
      {
        if (count < 0 || count > 32) { throw new InvalidDataException(); }
        uint value = 0;
        for (int position = 0; position < count;)
        {
          if (mRemainingBitCount > 0)
          {
            int take = Math.Min(count - position, mRemainingBitCount);
            // В C# сдвиг uint на 32 равен сдвигу на 0, поэтому маску строим вправо
            if (mCurrentBit != 0) { value |= (uint.MaxValue >> (32 - take)) << position; }
            mRemainingBitCount -= take;
            mLastBit = mCurrentBit;
            position += take;
          }
          else
          {
            value |= (uint)Bit() << position++;
          }
        }
        return value;
      }
      public static Bits Read(Input input, int depth = 0)
      {
        if (depth > 16) { throw new InvalidDataException(); }
        return input.Bit() == 0 ? (Bits)input : new RunBits(Read(input, depth + 1));
      }
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      public override int Bit()
      {
        if (mRemainingBitCount > 0) { mRemainingBitCount--; return mLastBit = mCurrentBit; }
        if (mFlipNextBit) { mCurrentBit = 1 - mCurrentBit; mFlipNextBit = false; return mLastBit = mCurrentBit; }
        mCurrentBit = mSource.Bit();
        if (mCurrentBit != mLastBit) { mSameBitCount = 1; }
        else if (++mSameBitCount >= mRunThreshold)
        {
          for (int bits = mInitialBitWidth; ; bits++)
          {
            if (bits > 30) { throw new InvalidDataException(); }
            int count = (int)mSource.ReadBits(bits);
            mRemainingBitCount = checked(mRemainingBitCount + count);
            if (count + 1 != 1 << bits) { break; }
          }
          mFlipNextBit = true;
          mSameBitCount = 0;
        }
        return mLastBit = mCurrentBit;
      }
    }

    private sealed class Codebook
    {
      private readonly Bits mSource;
      private readonly int[] mBitWidths, mNegativeBounds, mPositiveBounds, mTable = new int[8];
      private readonly int mFirstBucket;
      public Codebook(Bits source)
      {
        this.mSource = source;
        int count = (int)source.ReadBits(4);
        if (count == 0) { throw new InvalidDataException(); }
        mBitWidths = new int[count]; mNegativeBounds = new int[count]; mPositiveBounds = new int[count];
        for (int i = 0; i < count; i++)
        {
          int width = mBitWidths[i] = (int)source.ReadBits(i == count - 1 ? 5 : 4);
          if (i == 0 && width == 0) { continue; }
          if (width == 0 || width > 30) { throw new InvalidDataException(); }
          mNegativeBounds[i] = checked(-(1 << (width - 1)) + (i == 0 ? 0 : mNegativeBounds[i - 1]));
          mPositiveBounds[i] = checked((1 << (width - 1)) + (i == 0 ? -1 : mPositiveBounds[i - 1]));
        }
        mFirstBucket = (int)source.ReadBits(2);
        for (int i = 0; i < mTable.Length; i++) { mTable[i] = Signed(source.ReadBits(4), 4); }
      }
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      public int Read()
      {
        int bucket = mFirstBucket;
        if (bucket > 0)
        {
          for (int level = bucket - 1; level < 3; level++)
          {
            if (bucket > 3) { throw new InvalidDataException(); }
            int index = checked((int)mSource.ReadBits(bucket) + level * 2);
            if (index >= mTable.Length) { throw new InvalidDataException(); }
            bucket = mTable[index];
            if (bucket >= 0) { break; }
            bucket = -bucket;
          }
        }
        if (bucket >= mBitWidths.Length) { throw new InvalidDataException(); }
        int value = Signed(mSource.ReadBits(mBitWidths[bucket]), mBitWidths[bucket]);
        return bucket == 0 ? value : checked(value + (value < 0 ? mNegativeBounds[bucket - 1] : mPositiveBounds[bucket - 1] + 1));
      }
    }

    private static int Signed(uint value, int width) => width == 0 ? 0 : (int)(value << (32 - width)) >> (32 - width);
    private static float Finite(float value)
    {
      if (float.IsNaN(value) || float.IsInfinity(value)) { throw new InvalidDataException(); }
      return value;
    }

    private static void GetMinMax(float[] values, out float min, out float max)
    {
      if (values == null || values.Length == 0) { min = 0; max = 0; return; }
      float mn = values[0];
      float mx = values[0];
      for (int i = 1; i < values.Length; i++)
      {
        float v = values[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      min = mn;
      max = mx;
    }

    private struct Block
    {
      public int Start, Size;
      public float Offset, Scale;
      public Block(int start, int size, float offset, float scale)
      {
        Start = start;
        Size = size;
        Offset = offset;
        Scale = scale;
      }
    }

    private static float[] Samples(Input input, int count, double period)
    {
      float[] result = new float[count];
      var blocks = new List<Block>();
      float offset = 0, scale = 0;
      int compressed = 0;
      for (int start = 0; start < count; start += 256)
      {
        int size = Math.Min(256, count - start);
        if (input.Bit() != 0)
        {
          if (input.Bit() != 0) { offset = Finite(input.Reader.ReadSingle()); }
          if (input.Bit() != 0) { scale = Finite(input.Reader.ReadSingle()); }
          blocks.Add(new Block(start, size, offset, scale));
          compressed += size;
        }
        else
        {
          for (int j = 0; j < size; j++) { result[start + j] = Finite(input.Reader.ReadSingle()); }
        }
      }
      var book = new Codebook(RunBits.Read(input));
      int[] values = new int[compressed];
      int lag = checked((int)period);
      if (lag < 2) { throw new InvalidDataException(); }
      double x = 2 - (period - lag);
      // Предиктор использует кубическую интерполяцию предыдущего периода
      // Несжатые блоки в последовательность остатков не входят
      int initialCount = (int)Math.Min(values.Length, (long)lag + 3);
      for (int i = 0; i < initialCount; i++)
      {
        values[i] = book.Read();
      }
      for (int i = initialCount; i < values.Length; i++)
      {
        int residual = book.Read(), j = i - lag;
        double y0 = values[j - 2], y1 = values[j - 1], y2 = values[j], y3 = values[j + 1];
        double a2 = y2 - 2 * y1 + y0, a3 = y3 - 3 * y1 + 2 * y0;
        double cubic = (2 * a3 - 6 * a2) / 12, quadratic = 2 * a2 - a3 * 0.5;
        double linear = y1 - y0 - cubic - quadratic;
        // Порядок операций влияет на округление предиктора у полуцелых значений
        double prediction = cubic * x * x * x + quadratic * x * x + linear * x + y0;
        values[i] = checked(residual + (int)Math.Round(prediction, MidpointRounding.AwayFromZero));
      }
      int cursor = 0;
      for (int b = 0; b < blocks.Count; b++)
      {
        var block = blocks[b];
        int bStart = block.Start;
        int bSize = block.Size;
        float bOffset = block.Offset;
        float bScale = block.Scale;
        for (int j = 0; j < bSize; j++)
        {
          result[bStart + j] = Finite((float)((double)values[cursor++] * bScale + bOffset));
        }
      }
      return result;
    }

    private static Dictionary<string, List<Dictionary<string, object>>> Tables(Input input)
    {
      if (input.Reader.ReadUInt32() != 0x4442454b) { throw new NotSupportedException(); }
      int version = input.Reader.ReadInt32();
      if (version != 16 && version != 19 && version != 20) { throw new NotSupportedException(); }
      int count = input.Count(100);
      var tables = new Dictionary<string, List<Dictionary<string, object>>>();
      for (int table = 0; table < count; table++)
      {
        string name = input.Text();
        input.Text();
        int fields = input.Count(256);
        int[] types = new int[fields];
        string[] names = new string[fields];
        for (int i = 0; i < fields; i++) { types[i] = input.Reader.ReadInt32(); input.Value(types[i], true); }
        if (input.Count(256) != fields) { throw new InvalidDataException(); }
        for (int i = 0; i < fields; i++) { names[i] = input.Text(); }
        int rowCount = input.Count();
        var rows = new List<Dictionary<string, object>>(rowCount);
        for (int row = 0; row < rowCount; row++)
        {
          int size = input.Reader.ReadUInt16();
          if (size > input.Remaining) { throw new InvalidDataException(); }
          using (var record = new Input(input.Reader.ReadBytes(size)))
          {
            var values = new Dictionary<string, object>(fields);
            for (int field = 0; field < fields; field++) { values.Add(names[field], record.Value(types[field])); }
            if (record.Remaining != 0) { throw new InvalidDataException(); }
            rows.Add(values);
          }
        }
        int key = input.Reader.ReadInt32();
        if (key < -1 || key >= fields) { throw new InvalidDataException(); }
        tables.Add(name, rows);
      }
      int relations = input.Count(1024);
      for (int i = 0; i < relations * 4; i++) { input.Text(); }
      return tables;
    }

    private sealed class Node
    {
      public int Kind, Id;
      public string Name;
      public readonly List<Node> Children = new List<Node>();
    }

    private static Node ReadNode(Input input, int kind, int id, int depth = 0)
    {
      if (depth > 32) { throw new InvalidDataException(); }
      var node = new Node { Kind = kind, Id = id, Name = input.Text() };
      while (true)
      {
        int marker = input.Reader.ReadByte();
        if (marker == 0) { break; }
        if (marker != 1) { throw new InvalidDataException(); }
        int childKind = input.Reader.ReadByte();
        int childId = childKind == 4 ? 0 : input.Reader.ReadInt32();
        node.Children.Add(ReadNode(input, childKind, childId, depth + 1));
      }
      return node;
    }

    private static void Names(Node node, string path, Dictionary<long, Tuple<string, string>> names)
    {
      long key = (long)node.Kind << 32 | (uint)node.Id;
      if (node.Kind == 5)
      {
        string phase = (node.Name == "a" || node.Name == "ab") ? "A" : (node.Name == "b" || node.Name == "bc") ? "B" : (node.Name == "c" || node.Name == "ca") ? "C" : "N";
        string name = path + node.Name;
        if (node.Name == "o" && path.Length > 0) { name = path.Insert(path.Length - 1, "3") + node.Name; }
        if (!names.ContainsKey(key)) { names.Add(key, Tuple.Create(name.TrimEnd(), phase)); }
      }
      else if (node.Kind == 6) { if (!names.ContainsKey(key)) { names.Add(key, Tuple.Create(path + node.Name, "")); } }
      else
      {
        string prefix = path + node.Name + (node.Kind == 0 ? "" : " ");
        foreach (Node child in node.Children) { Names(child, prefix, names); }
      }
    }

    private sealed class Analog
    {
      public int Id;
      public double Start, Rate;
      public float[] Values;
    }
    private struct DigitalEvent
    {
      public int Time, Value;
      public DigitalEvent(int time, int value) { Time = time; Value = value; }
    }
    private sealed class Digital
    {
      public uint Id;
      public readonly List<DigitalEvent> Events = new List<DigitalEvent>();
    }

    private static uint[] PackDigital(int samples, double rate, List<DigitalEvent> events)
    {
      uint[] words = new uint[(samples + 31) / 32];
      // При частых переключениях линейный проход дешевле поиска каждой границы
      if (events.Count > Math.Max(1, samples / 32))
      {
        int current = 0;
        for (int sample = 0; sample < samples; sample++)
        {
          while (current + 1 < events.Count && events[current + 1].Time <= sample * 1000.0 / rate) current++;
          if (events[current].Value != 0) words[sample >> 5] |= 1u << (sample & 31);
        }
        return words;
      }
      int start = 0;
      for (int e = 0; e < events.Count && start < samples; e++)
      {
        int end = samples;
        if (e + 1 < events.Count)
        {
          int left = start, right = samples;
          // Сохраняем точное сравнение исходного декодера, без округления времени события
          while (left < right)
          {
            int middle = left + (right - left) / 2;
            if (middle * 1000.0 / rate < events[e + 1].Time) left = middle + 1;
            else right = middle;
          }
          end = left;
        }
        if (events[e].Value != 0 && start < end)
        {
          int first = start >> 5, last = (end - 1) >> 5;
          uint firstMask = uint.MaxValue << (start & 31);
          uint lastMask = uint.MaxValue >> (31 - ((end - 1) & 31));
          if (first == last) words[first] |= firstMask & lastMask;
          else
          {
            words[first] |= firstMask;
            for (int word = first + 1; word < last; word++) words[word] = uint.MaxValue;
            words[last] |= lastMask;
          }
        }
        start = end;
      }
      return words;
    }

    private static int Int(Dictionary<string, object> row, string field) => (int)row[field];
    private static string Clean(string text) => text.Replace('.', ' ').Replace(',', '.').Replace('\r', ' ').Replace('\n', ' ');
    private static string G(double value) => BinaryDecimal.Round(value, 6).ToString("G6", mInvariant);



    public static OscillogramData Read(string sourcePath)
    {
      byte[] bytes = File.ReadAllBytes(sourcePath);
      if (bytes.Length < 276 || bytes[263] != 4) { throw new NotSupportedException(); }
      Dictionary<string, List<Dictionary<string, object>>> tables;
      byte[] stream;
      using (var input = new Input(bytes))
      {
        input.Reader.BaseStream.Position = 264;
        tables = Tables(input);
        stream = input.Blob();
        if (input.Remaining != 0) { throw new InvalidDataException(); }
      }
      var common = tables["CommonParameters"].Single();
      Node tree;
      using (var input = new Input((byte[])common["NameTree"]))
      {
        if (input.Reader.ReadByte() != 4) { throw new NotSupportedException(); }
        tree = ReadNode(input, 0, 0);
        if (input.Remaining != 1 || input.Reader.ReadByte() != 0) { throw new InvalidDataException(); }
      }
      var names = new Dictionary<long, Tuple<string, string>>();
      foreach (Node child in tree.Children) { Names(child, "", names); }
      var analogs = new List<Analog>();
      var digitals = new List<Digital>();
      double timestamp;
      using (var input = new Input(stream))
      {
        int version = input.Reader.ReadInt32();
        if (version != 6) { throw new NotSupportedException(); }
        timestamp = input.Reader.ReadDouble();
        input.Reader.ReadByte();
        double frequency = Finite(input.Reader.ReadSingle());
        Finite(input.Reader.ReadSingle());
        if (frequency <= 0) { throw new InvalidDataException(); }
        int digitalCount = input.Count(65536);
        for (int i = 0; i < digitalCount; i++)
        {
          var digital = new Digital { Id = input.Reader.ReadUInt32() };
          digital.Events.Add(new DigitalEvent(0, input.Bit()));
          int time = 0;
          while (input.Bit() != 0)
          {
            int value = input.Bit(), width = input.Bit() != 0 ? 8 : 32;
            time = checked(time + checked((int)input.ReadBits(width)));
            digital.Events.Add(new DigitalEvent(time, value));
            if (digital.Events.Count > 1000000) { throw new InvalidDataException(); }
          }
          digitals.Add(digital);
        }
        int analogCount = input.Count(4096);
        long totalSamples = 0;
        for (int i = 0; i < analogCount; i++)
        {
          var analog = new Analog { Start = input.Reader.ReadDouble(), Rate = Finite(input.Reader.ReadSingle()), Id = input.Reader.ReadInt32() };
          int count = input.Reader.ReadInt32();
          totalSamples += count;
          if (count <= 0 || count > 10000000 || totalSamples > 64000000 || analog.Rate <= 0 || double.IsNaN(analog.Start) || double.IsInfinity(analog.Start)) { throw new InvalidDataException(); }
          analog.Values = Samples(input, count, analog.Rate / frequency);
          analogs.Add(analog);
        }
        if (input.Remaining != 0 || analogs.Count == 0) { throw new InvalidDataException(); }
      }
      int samples = analogs[0].Values.Length;
      double rate = analogs[0].Rate;
      if (analogs.Any(a => a.Values.Length != samples)) { throw new NotSupportedException(); }
      var analogRowsById = tables["AnalogChannels"].ToDictionary(r => Int(r, "ChannelID"));
      var typesById = tables["Types"].ToDictionary(r => Int(r, "TypeID"));
      var channels = new List<AnalogChannel>(analogs.Count);
      for (int i = 0; i < analogs.Count; i++)
      {
        var analog = analogs[i];
        if (!analogRowsById.TryGetValue(analog.Id, out var row)) { throw new InvalidDataException(); }
        if (!typesById.TryGetValue(Int(row, "TypeID"), out var type)) { throw new InvalidDataException(); }
        string unit = (string)type["TypeEdinShort"];
        if (unit == "В") { unit = "V"; }
        if (unit == "А") { unit = "A"; }
        var name = names[5L << 32 | (uint)analog.Id];
        GetMinMax(analog.Values, out float min, out float max);
        // Поток содержит физические float-значения — при чтении не квантуем их в целый DAT
        channels.Add(new AnalogChannel(i + 1, Clean(name.Item1).Trim(), name.Item2, "", unit,
          1, 0, analog.Start * 1000000, min, max, 1, 1, SignalType.Secondary));
      }
      var discreteRowsByKey = tables["DiscreteChannels"].ToDictionary(r => ((uint)Int(r, "Registrator") | (uint)Int(r, "Kanal") << 10));
      var discreteChannels = new List<DiscreteChannel>(digitals.Count);
      for (int i = 0; i < digitals.Count; i++)
      {
        uint key = digitals[i].Id & 0x03ffffff;
        if (!discreteRowsByKey.TryGetValue(key, out var row)) { throw new InvalidDataException(); }
        discreteChannels.Add(new DiscreteChannel(i + 1, Clean(names[6L << 32 | (uint)Int(row, "ChannelID")].Item1).Trim(), "", "", false));
      }
      DateTime date = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc).AddTicks(checked((long)(Math.Floor(timestamp * 100) * 100000))).ToLocalTime();
      var configuration = new ConfigurationHandler
      {
        StationName = Clean(tree.Name).Trim(),
        DeviceId = "1",
        Version = ComtradeVersion.V1999,
        AnalogChannels = channels,
        DiscreteChannels = discreteChannels,
        SampleRates = new List<SampleRate> { new SampleRate(rate, samples) },
        Frequency = 50,
        StartTime = date,
        TriggerTime = date,
        DataFileType = DataFileType.ASCII,
        TimeMultiplicationFactor = 1
      };
      uint[][] discreteData = new uint[digitals.Count][];
      Parallel.For(0, digitals.Count, channel => discreteData[channel] = PackDigital(samples, rate, digitals[channel].Events));
      long[] timestamps = new long[samples];
      for (int i = 0; i < timestamps.Length; i++)
        timestamps[i] = (long)Math.Round(i * 1000000.0 / rate, MidpointRounding.AwayFromZero);
      return NativeSampleBuilder.CreatePacked(configuration, analogs.Select(channel => channel.Values).ToArray(), discreteData, timestamps);
    }
  }
}
