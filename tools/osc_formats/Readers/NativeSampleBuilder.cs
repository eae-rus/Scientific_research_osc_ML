namespace OscFormats.Native
{
  using OscFormats.Model;
  using System;
  using System.Text;
  using System.Threading.Tasks;

  // Завершает декодирование: калибрует аналоги и упаковывает дискреты, не сохраняя промежуточную запись
  internal static class NativeSampleBuilder
  {
    internal static OscillogramData CreatePacked(ConfigurationHandler configuration, float[][] analog, uint[][] digital, long[] timestamps)
    {
      var time = BuildTime(configuration, timestamps.Length, timestamps);
      var channels = new ChannelSamples[analog.Length];
      Parallel.For(0, analog.Length, c =>
      {
        var metadata = configuration.AnalogChannels[c];
        double[] values = new double[time.Count];
        for (int i = 0; i < values.Length; i++) values[i] = analog[c][i] * metadata.A + metadata.B;
        channels[c] = ChannelSamples.TakeAnalog(time, values);
      });
      var discrete = new ChannelSamples[digital.Length];
      for (int c = 0; c < discrete.Length; c++) discrete[c] = ChannelSamples.TakeDigital(time, digital[c]);
      return Finish(configuration, time, channels, discrete);
    }

    internal static OscillogramData Create(ConfigurationHandler configuration, float[][] analog, bool[][] digital, long[] timestamps = null)
    {
      int count = analog.Length > 0 ? analog[0].Length : digital.Length > 0 ? digital[0].Length : timestamps?.Length ?? 0;
      var time = BuildTime(configuration, count, timestamps);
      var channels = new ChannelSamples[analog.Length];
      Parallel.For(0, analog.Length, c =>
      {
        var metadata = configuration.AnalogChannels[c];
        double[] values = new double[count];
        for (int i = 0; i < count; i++) values[i] = analog[c][i] * metadata.A + metadata.B;
        channels[c] = ChannelSamples.TakeAnalog(time, values);
      });
      return Finish(configuration, time, channels, digital);
    }

    internal static OscillogramData Create(ConfigurationHandler configuration, double[][] analog, bool[][] digital, long[] timestamps = null)
    {
      int count = analog.Length > 0 ? analog[0].Length : digital.Length > 0 ? digital[0].Length : timestamps?.Length ?? 0;
      var time = BuildTime(configuration, count, timestamps);
      var channels = new ChannelSamples[analog.Length];
      Parallel.For(0, analog.Length, c =>
      {
        var metadata = configuration.AnalogChannels[c];
        double[] values = analog[c];
        for (int i = 0; i < count; i++) values[i] = values[i] * metadata.A + metadata.B;
        channels[c] = ChannelSamples.TakeAnalog(time, values);
      });
      return Finish(configuration, time, channels, digital);
    }

    private static OscillogramData Finish(ConfigurationHandler configuration, TimeAxis time, ChannelSamples[] analog, bool[][] digital)
    {
      var channels = new ChannelSamples[digital.Length];
      Parallel.For(0, digital.Length, c => channels[c] = ChannelSamples.FromDigital(time, digital[c]));
      return Finish(configuration, time, analog, channels);
    }

    private static OscillogramData Finish(ConfigurationHandler configuration, TimeAxis time, ChannelSamples[] analog, ChannelSamples[] channels)
    {
      configuration.Encoding = configuration.Encoding ?? Encoding.UTF8;
      configuration.AnalogChannelsCount = analog.Length;
      configuration.DiscreteChannelsCount = channels.Length;
      configuration.ChannelsCount = analog.Length + channels.Length;
      configuration.SamplingRateCount = configuration.SampleRates.Count;
      return new OscillogramData(configuration, time, analog, channels);
    }

    private static TimeAxis BuildTime(ConfigurationHandler configuration, int count, long[] timestamps)
    {
      double[] values = new double[count];
      var rates = configuration.SampleRates;
      if (timestamps == null)
      {
        timestamps = new long[count];
        double rate = rates.Count > 0 ? rates[0].SamplingFrequency : 0;
        if (rate > 0)
          for (int i = 0; i < count; i++) timestamps[i] = (long)(i * 1000000.0 / rate);
      }
      if (rates.Count == 0 || Math.Abs(rates[0].SamplingFrequency) < 0.01)
      {
        double factor = configuration.TimeMultiplicationFactor * (configuration.TimeLineNanoSecondResolution ? 1e-6 : 1e-3);
        for (int i = 0; i < count; i++) values[i] = (timestamps?[i] ?? 0) * factor;
      }
      else
      {
        double time = 0;
        int segment = 0;
        double step = 1e9 / rates[0].SamplingFrequency;
        for (int i = 0; i < count; i++)
        {
          values[i] = time * 1e-6;
          if (i >= rates[segment].LastSampleNumber && segment + 1 < rates.Count)
            step = 1e9 / rates[++segment].SamplingFrequency;
          time += step;
        }
      }
      double recordedUnit = configuration.TimeMultiplicationFactor * (configuration.TimeLineNanoSecondResolution ? 1e-6 : 1e-3);
      return TimeAxis.TakeOwnership(values, timestamps, recordedUnit);
    }
  }
}
