using OscFormats.Model;
using OscFormats.Native;
using System.Globalization;
using System.Text;
using System.Text.Json;

// Протокол версии 1: метаданные JSON, затем время float64 в секундах
// с порядком байтов от младшего к старшему, откалиброванные аналоги float64
// и дискреты uint8. Отсчёты аналогов и дискретов сгруппированы по каналам.
internal static class Program
{
    private static int Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
        try
        {
            if (args.Length != 4 && args.Length != 5)
                throw new ArgumentException("format source output-directory recover-incomplete [wconfig]");
            string source = Path.GetFullPath(args[1]);
            bool recover = bool.Parse(args[3]);
            OscillogramData record = args[0] switch
            {
                "parma" => ParmaDoConverter.Read(source),
                "ekra-dfr" => EkraDfrReader.Read(source),
                "ekra-ndr" => EkraNdrReader.Read(source, args.Length == 5 ? args[4] : throw new ArgumentException("WConfig required")),
                "bresler" => BreslerReader.Read(source),
                "blackbox" => BlackBoxReader.Read(source),
                "neva" => NevaReader.Read(source, recover),
                "res3" => Res3Reader.Read(source, recover),
                _ => throw new NotSupportedException(args[0])
            };
            var cfg = record.Configuration;
            Directory.CreateDirectory(args[2]);
            using (var writer = new BinaryWriter(File.Create(Path.Combine(args[2], "samples.bin"))))
            {
                for (int i = 0; i < record.Time.Count; i++) writer.Write(record.Time[i] / 1000.0);
                foreach (var channel in record.Analog)
                    for (int i = 0; i < record.Time.Count; i++) writer.Write(channel[i]);
                foreach (var channel in record.Digital)
                    for (int i = 0; i < record.Time.Count; i++) writer.Write((byte)(channel[i] != 0 ? 1 : 0));
            }
            var metadata = new
            {
                protocol = 1, format = args[0], total_samples = record.Time.Count,
                station_name = cfg.StationName, rec_dev_id = cfg.DeviceId,
                frequency = cfg.Frequency, start_timestamp = cfg.StartTime, trigger_timestamp = cfg.TriggerTime,
                sample_rates = cfg.SampleRates.Select(r => new double[] { r.SamplingFrequency, r.LastSampleNumber }),
                analog_channels = cfg.AnalogChannels.Select(c => new {
                    n = c.Number, name = c.Name, ph = c.Phase, ccbm = c.CircuitComponent,
                    uu = c.Unit, a = c.A, b = c.B, skew = c.Skew, min = c.Min, max = c.Max,
                    primary = c.Primary, secondary = c.Secondary,
                    pors = c.SignalType == SignalType.Primary ? "P" : "S"
                }),
                status_channels = cfg.DiscreteChannels.Select(c => new {
                    n = c.Number, name = c.Name, ph = c.Phase, ccbm = c.CircuitComponent, y = c.NormalState ? 1 : 0
                })
            };
            File.WriteAllText(Path.Combine(args[2], "metadata.json"), JsonSerializer.Serialize(metadata));
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine($"{error.GetType().Name}: {error.Message}");
            return 1;
        }
    }
}
