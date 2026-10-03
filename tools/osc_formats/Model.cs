// Минимальная внутренняя модель хранения данных для декодеров форматов производителей.
// Не требует парсера или средства записи COMTRADE, неявного пересчёта масштабов,
// графического интерфейса или среды выполнения программ производителей.
namespace OscFormats.Model;

internal enum SignalType { Primary, Secondary }
internal enum ComtradeVersion { V1999, V2013 }
internal enum DataFileType { ASCII, BINARY }

internal sealed class AnalogChannel
{
    public int Number { get; }
    public string Name { get; }
    public string Phase { get; }
    public string CircuitComponent { get; }
    public string Unit { get; }
    public double A { get; }
    public double B { get; }
    public double Skew { get; }
    public double Min { get; }
    public double Max { get; }
    public double Primary { get; }
    public double Secondary { get; }
    public SignalType SignalType { get; }

    public AnalogChannel(int number, string name, string phase, string circuitComponent,
        string unit, double a, double b, double skew, double min, double max,
        double primary, double secondary, SignalType signalType)
    {
        Number = number; Name = name; Phase = phase; CircuitComponent = circuitComponent;
        Unit = unit; A = a; B = b; Skew = skew; Min = min; Max = max;
        Primary = primary; Secondary = secondary; SignalType = signalType;
    }
}

internal sealed class DiscreteChannel
{
    public int Number { get; }
    public string Name { get; }
    public string Phase { get; }
    public string CircuitComponent { get; }
    public bool NormalState { get; }
    public DiscreteChannel(int number, string name, string phase, string circuitComponent, bool normalState)
    {
        Number = number; Name = name; Phase = phase;
        CircuitComponent = circuitComponent; NormalState = normalState;
    }
}

internal sealed record SampleRate(double SamplingFrequency, int LastSampleNumber);

internal sealed class ConfigurationHandler
{
    public string StationName { get; set; }
    public string DeviceId { get; set; }
    public ComtradeVersion Version { get; set; }
    public List<AnalogChannel> AnalogChannels { get; set; }
    public List<DiscreteChannel> DiscreteChannels { get; set; }
    public List<SampleRate> SampleRates { get; set; }
    public double Frequency { get; set; }
    public DateTime StartTime { get; set; }
    public DateTime TriggerTime { get; set; }
    public DataFileType DataFileType { get; set; }
    public double TimeMultiplicationFactor { get; set; }
    public bool TimeLineNanoSecondResolution { get; set; }
    public System.Text.Encoding Encoding { get; set; }
    public int AnalogChannelsCount { get; set; }
    public int DiscreteChannelsCount { get; set; }
    public int ChannelsCount { get; set; }
    public int SamplingRateCount { get; set; }
}

internal sealed class TimeAxis
{
    private readonly double[] values;
    public long[] RecordedTimestamps { get; }
    public double RecordedUnit { get; }
    public int Count => values.Length;
    public double this[int index] => values[index];
    private TimeAxis(double[] values, long[] timestamps, double unit)
    { this.values = values; RecordedTimestamps = timestamps; RecordedUnit = unit; }
    public static TimeAxis TakeOwnership(double[] values, long[] timestamps, double unit)
    {
        if (values.Length != timestamps.Length) throw new InvalidDataException("Time axis length mismatch");
        return new TimeAxis(values, timestamps, unit);
    }
}

internal sealed class ChannelSamples
{
    private readonly double[] analog;
    private readonly uint[] digital;
    private ChannelSamples(double[] analog, uint[] digital) { this.analog = analog; this.digital = digital; }
    public double this[int index] => analog != null ? analog[index] : (digital[index / 32] >> (index % 32)) & 1;
    public static ChannelSamples TakeAnalog(TimeAxis time, double[] values)
    {
        if (values.Length != time.Count) throw new InvalidDataException("Analog length mismatch");
        return new ChannelSamples(values, null);
    }
    public static ChannelSamples TakeDigital(TimeAxis time, uint[] values)
    {
        if (values.Length != (time.Count + 31) / 32) throw new InvalidDataException("Digital length mismatch");
        return new ChannelSamples(null, values);
    }
    public static ChannelSamples FromDigital(TimeAxis time, bool[] values)
    {
        if (values.Length != time.Count) throw new InvalidDataException("Digital length mismatch");
        var packed = new uint[(values.Length + 31) / 32];
        for (int i = 0; i < values.Length; i++) if (values[i]) packed[i / 32] |= 1u << (i % 32);
        return TakeDigital(time, packed);
    }
}

internal sealed record OscillogramData(ConfigurationHandler Configuration, TimeAxis Time,
    ChannelSamples[] Analog, ChannelSamples[] Digital);
