using OscFormats.Model;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;

namespace OscFormats.Native
{
  /// <summary>
  /// Читатель осциллограмм ПАРМА (.do с продолжениями .d01, .d02, ...) без внешних утилит.
  /// Порт алгоритма разбора программы «Транскоп» (ООО «ПАРМА», transcop_pack.js).
  ///
  /// Формат файла (всё little-endian, строки в CP866):
  /// - байты 0..2   — версия формата цифрами (поддерживается от 40);
  /// - 24..151      — комментарий (ASCIIZ);
  /// - 152..232     — имя прибора (ASCIIZ);
  /// - 233/237      — время начала записи: uint32 секунды / uint32 микросекунды;
  /// - 241/245      — момент пуска: uint32 секунды / uint32 микросекунды;
  /// - 249          — uint32 длительность части (для продолжений .d01...);
  /// - 253          — uint16 число каналов;
  /// - 255          — int16 часовой пояс (часы/минуты/секунды — по величине);
  /// - 259          — uint16 смещение первой записи канала.
  ///
  /// Запись канала: uint16 тип (0 — аналоговый, 1 — дискретный, 2 — блок ОМП, 3 — пропуск),
  /// имя (66 байт, до ':' — присоединение), uint32 смещение данных, uint32 длина данных.
  /// У аналогового канала далее: частота дискретизации и коэффициенты (float32), единица
  /// измерения (8 байт), метод сжатия (0 — несжатые int16, 3 — дельта-сжатие упакованными
  /// 4-битными приращениями), смещение нуля, номинал. У дискретного: uint32 число событий,
  /// события по 5 байт (состояние + uint32 время, мкс от начала записи).
  ///
  /// Read возвращает запись в памяти; отдельный экспорт создаёт COMTRADE 2013 (ASCII)
  /// Значения каналов пишутся во вторичных величинах с масштабом a = (|макс|+|мин|)/5000
  /// и смещением b = смещение нуля — как в экспорте «Транскопа».
  /// </summary>
  internal static class ParmaDoConverter
  {
    /// <summary>Число служебных точек в начале массива данных «Транскопа» (transcop.headLength).</summary>
    private const int HeadLength = 20;
    private const double BaseFrequency = 50;

    /// <summary>Признаки фаз в имени канала (transcop.UI) и соответствующие фазы (transcop.Ph).</summary>
    private static readonly string[][] mPhaseNamePatterns =
    {
      new[] { "UAB", "VAB", "IAB" },
      new[] { "UBC", "VBC", "IBC" },
      new[] { "UCA", "VCA", "ICA" },
      new[] { "UA", "VA", "V_A", "IA", "I_A" },
      new[] { "UB", "VB", "V_B", "IB", "I_B" },
      new[] { "UC", "VC", "V_C", "IC", "I_C" },
      new[] { "U0", "3U0", "V0", "3V0", "I0", "3I0", "UO", "3UO", "VO", "3VO", "IO", "3IO" }
    };

    private static readonly string mNumericSuffixChars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    /// <summary>Аналоговый канал осциллограммы.</summary>
    private sealed class AnalogData
    {
      public string Name = "";
      public string CircuitComponent = "";
      public string Phase = "";
      public string Unit = "";
      public double K1;              // первичный коэффициент (идёт в cfg как primary)
      public double K2, K7;          // множители сырых отсчётов
      public double ZeroOffset;      // смещение нуля (в cfg — b)
      public double Nominal;         // номинальный размах
      public double MaxPeak;         // максимум локальных пиков
      public double MinPeak;         // минимум локальных пиков
      public double MaxAmplitude = double.NaN;
      public int Method;             // 0 — несжатые int16, 3 — дельта-сжатие
      public readonly List<double> Samples = new List<double>();
    }

    /// <summary>Дискретный канал осциллограммы.</summary>
    private sealed class DiscreteData
    {
      public string Name = "";
      public string CircuitComponent = "";
      public int InitialState;
      /// <summary>Времена событий, мкс от начала записи (первое событие учтено в InitialState).</summary>
      public readonly List<double> EventTimes = new List<double>();

      public int ValueAt(double timeUs)
      {
        int value = InitialState;
        // Первое событие уже учтено в начальном состоянии — как в transcop.doDiscrVal.
        for (int i = 1; i < EventTimes.Count && EventTimes[i] < timeUs; i++)
        {
          value ^= 1;
        }
        return value;
      }
    }

    /// <summary>
    /// Конвертирует осциллограмму ПАРМА в COMTRADE (пара .cfg/.dat в <paramref name="outputDirectory"/>).
    /// Возвращает путь к созданному .cfg. Файлы-продолжения .d01, .d02, ... берутся из того же каталога.
    /// </summary>


    public static OscillogramData Read(string sourceFilePath)
    {
      var record = new ParmaRecord();
      record.Read(sourceFilePath);
      return CreateRecord(record);
    }

    // ===== Разбор формата ==============================================================

    private sealed class ParmaRecord
    {
      public string InstrumentName = "";
      public double BeginTimeUs;    // начало записи (с учётом пояса)
      public double TriggerTimeUs;  // момент пуска (с учётом пояса)
      public double Sampling = 19200;
      /// <summary>Шкала времени точек, мкс; первые HeadLength значений — служебный резерв.</summary>
      public readonly List<double> Timeline = new List<double>();
      public readonly List<AnalogData> Analogs = new List<AnalogData>();
      public readonly List<DiscreteData> Discretes = new List<DiscreteData>();

      private static int U16(byte[] data, int pos) { return BitConverter.ToUInt16(data, pos); }
      private static int I16(byte[] data, int pos) { return BitConverter.ToInt16(data, pos); }
      private static double U32(byte[] data, int pos) { return BitConverter.ToUInt32(data, pos); }
      private static double F32(byte[] data, int pos) { return BitConverter.ToSingle(data, pos); }

      /// <summary>Суммарная длительность всех частей до текущей (oscils.readlg), мкс.</summary>
      private double mPreviousPartsUs;
      private string mCurrentExtension = "do";

      public void Read(string sourceFilePath)
      {
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
        var oem866 = Encoding.GetEncoding(866);

        string baseDirectory = Path.GetDirectoryName(sourceFilePath);
        string stem = Path.GetFileNameWithoutExtension(sourceFilePath);

        string partPath = sourceFilePath;
        bool longOscil = false;

        while (partPath != null)
        {
          // Индексы каналов сбрасываются на каждой части (codecs.readDO: tmpAnalogs = 0),
          // данные продолжения дописываются в уже существующие каналы.
          int analogIndex = 0, discreteIndex = 0;
          byte[] data = File.ReadAllBytes(partPath);
          int bufferEnd = data.Length;

          // --- Заголовок части ---
          var versionChars = new StringBuilder();
          for (int i = 0; i < 3; i++)
          {
            if (data[i] > 0)
            {
              versionChars.Append(oem866.GetString(data, i, 1));
            }
          }
          int version;
          if (!int.TryParse(versionChars.ToString(), NumberStyles.Integer, CultureInfo.InvariantCulture, out version) || version < 40)
          {
            throw new InvalidOperationException(
              $"Файл ПАРМА '{Path.GetFileName(partPath)}' имеет неподдерживаемую версию формата ({versionChars}).");
          }

          double partLengthUs = U32(data, 249);
          int firstRecordOffset = U16(data, 259);

          if (!longOscil)
          {
            InstrumentName = ReadAsciiz(data, 152, 233, oem866);
            double timeUs = ReadPackedTime(data, 233);
            double dateUs = ReadPackedTime(data, 241);
            int timeZone = I16(data, 255);
            if (timeZone < 0 && Math.Abs(timeZone) > 25)
            {
              timeZone = Math.Abs(timeZone);
            }
            double timeZoneUs = Math.Abs(timeZone) < 25
              ? timeZone * 3600000000.0
              : Math.Abs(timeZone) < 3600 ? timeZone * 60000000.0 : timeZone * 1000000.0;
            BeginTimeUs = timeUs + timeZoneUs;
            TriggerTimeUs = dateUs + timeZoneUs;
            for (int i = 0; i < HeadLength; i++)
            {
              Timeline.Add(0);
            }
          }

          // --- Записи каналов (codecs.doChanels) ---
          int pos = firstRecordOffset;
          int currentNum = longOscil ? Timeline.Count : HeadLength;
          while (pos < bufferEnd)
          {
            int recordStart = pos;
            int type = U16(data, pos);
            pos += 2;

            if (type == 2)
            {
              // Блок ОМП замыкает файл (данные ОМП «Транскопом» при конвертации не используются).
              break;
            }

            string name, circuitComponent;
            ReadChannelName(data, pos, oem866, out name, out circuitComponent);
            pos += 66;

            int dataOffset = (int)U32(data, pos);
            pos += 4;
            int dataLength = (int)U32(data, pos);
            pos += 4;

            if (type == 1)
            {
              DiscreteData channel = longOscil ? Discretes[discreteIndex++] : CreateDiscrete(name, circuitComponent);

              int eventCount = U16(data, pos) | (U16(data, pos + 2) << 16);
              int normalState = data[pos + 4];
              if (eventCount == 0)
              {
                channel.Name = "Error: " + channel.Name;
              }

              int eventPos = recordStart + dataOffset;
              for (int h = 0; h < eventCount; h++)
              {
                if (h == 0 && longOscil)
                {
                  // Первое событие продолжения повторяет последнее из предыдущей части.
                  eventPos += 5;
                  continue;
                }
                if (h == 0)
                {
                  channel.InitialState = normalState == data[eventPos] ? 0 : 1;
                }
                eventPos++;
                channel.EventTimes.Add(mPreviousPartsUs + U32(data, eventPos));
                eventPos += 4;
              }
              pos = recordStart + dataOffset + dataLength;
            }
            else if (type == 0)
            {
              AnalogData channel;
              if (longOscil)
              {
                channel = Analogs[analogIndex++];
                channel.Method = U16(data, pos + 34);
              }
              else
              {
                channel = new AnalogData
                {
                  Name = name,
                  CircuitComponent = circuitComponent,
                  K1 = F32(data, pos + 10),
                  K2 = F32(data, pos + 14),
                  K7 = F32(data, pos + 18),
                  ZeroOffset = F32(data, pos + 36) / F32(data, pos + 10),
                  Nominal = U16(data, pos + 40) * F32(data, pos + 14) * F32(data, pos + 18) +
                            F32(data, pos + 36) / F32(data, pos + 10),
                  Unit = ReadUnit(data, pos + 26, oem866),
                  Method = U16(data, pos + 34),
                  MaxPeak = 0,
                  MinPeak = 0
                };
                channel.MinPeak = channel.Nominal;
                Analogs.Add(channel);
                analogIndex++;
                Sampling = F32(data, pos + 4);
              }
              channel.Phase = DetectPhase(channel.Name);
              double samplingStepUs = 1000000.0 / Sampling;

              ReadAnalogSamples(channel, data, recordStart + dataOffset, recordStart + dataOffset + dataLength,
                bufferEnd, currentNum, samplingStepUs);

              // Итоговые амплитуды (transcop.doChanels, конец ветки 0==c)
              if (channel.MaxPeak == 0)
              {
                channel.MaxPeak = channel.Nominal;
              }
              if (double.IsNaN(channel.MaxAmplitude) || Math.Abs(channel.MaxAmplitude) < Math.Abs(channel.Nominal))
              {
                channel.MaxAmplitude = channel.Nominal;
                channel.Nominal = 2 * channel.MaxPeak;
              }
              pos = recordStart + dataOffset + dataLength;
            }
            else if (type == 3)
            {
              pos = recordStart + dataOffset + dataLength;
            }
            else
            {
              throw new InvalidOperationException($"Неизвестный тип записи канала ПАРМА: {type}.");
            }
          }

          // --- codecs.doEnding: поиск продолжения имя.d01, .d02 ... ---
          string nextExtension = NextPartExtension(mCurrentExtension);
          string candidate = Path.Combine(baseDirectory, stem + "." + nextExtension);
          if (File.Exists(candidate))
          {
            longOscil = true;
            mPreviousPartsUs += partLengthUs;
            mCurrentExtension = nextExtension;
            partPath = candidate;
          }
          else
          {
            partPath = null;
          }
        }

        AdjustFrequencyChannels();
      }

      private DiscreteData CreateDiscrete(string name, string circuitComponent)
      {
        var channel = new DiscreteData { Name = name, CircuitComponent = circuitComponent };
        Discretes.Add(channel);
        return channel;
      }

      /// <summary>Чтение отсчётов аналогового канала (codecs.doChanels, методы 0 и 3).</summary>
      private void ReadAnalogSamples(AnalogData channel, byte[] data, int dataPos, int dataEnd, int bufferEnd,
        int currentNum, double samplingStepUs)
      {
        bool fillsTimeline = ReferenceEquals(Analogs[0], channel);
        if (channel.Method == 0)
        {
          // Несжатые отсчёты int16
          int h = currentNum;
          for (; dataPos < dataEnd; h++)
          {
            double value = dataPos + 2 <= bufferEnd ? BitConverter.ToInt16(data, dataPos) : 1;
            dataPos += 2;
            WriteSample(channel, h, value * channel.K2 * channel.K7);
            if (fillsTimeline)
            {
              Timeline.Add(Timeline[Timeline.Count - 1] + samplingStepUs);
            }
            SearchMaxMin(channel, h);
          }
        }
        else if (channel.Method == 3)
        {
          // Дельта-сжатие: int32, число «сырых» отсчётов, затем упакованные приращения
          dataPos += 4;
          int window = I16(data, dataPos);
          dataPos += 2;

          int h = currentNum;
          int rawEnd = h + window;
          for (; h < rawEnd && dataPos + 2 <= bufferEnd; h++)
          {
            WriteSample(channel, h, BitConverter.ToInt16(data, dataPos));
            dataPos += 2;
            if (fillsTimeline)
            {
              Timeline.Add(Timeline[Timeline.Count - 1] + samplingStepUs);
            }
          }

          while (dataPos < dataEnd)
          {
            int control = data[dataPos];
            dataPos++;
            for (int nibbleIndex = 2; nibbleIndex < 4; nibbleIndex++)
            {
              int nibble = nibbleIndex == 2 ? control & 15 : control >> 4;
              if (nibble == 0)
              {
                // Повтор окна: oscils[d].slice(n, h) + concat — копия добавляется
                // В КОНЕЦ массива из окна [h-window..h-1]. У «Транскопа» h после
                // обрыва предыдущего блока (break без h++) остаётся на последней
                // записанной точке, поэтому источник и цель смещены относительно
                // наивного «скопировать в h». Шкала времени (cnt) при повторе
                // не расширяется — «хвост» перезаписывается следующей частью.
                int sourceStart = h - window;
                int appendStart = HeadLength + channel.Samples.Count;
                for (int i = 0; i < window; i++, h++)
                {
                  WriteSample(channel, appendStart + i, SampleAt(channel, sourceStart + i));
                }
                continue;
              }

              int threshold = 1 << nibble;
              int bits = nibble + 1;
              int period = 1 << bits;
              int packedEnd = dataPos + (bits * window + 7) / 8;
              uint mask = (uint)period - 1;
              uint queue = 0;
              int remaining = 32;
              for (int w = 0; w < window; w++, h++)
              {
                if (dataPos < packedEnd)
                {
                  while (remaining >= 8)
                  {
                    int shift = 32 - remaining;
                    remaining -= 8;
                    queue += (uint)(dataPos < bufferEnd ? data[dataPos] : 0) << shift;
                    dataPos++;
                  }
                }
                uint delta = queue & mask;
                double baseValue = SampleAt(channel, h - window);
                WriteSample(channel, h, delta >= (uint)threshold
                  ? baseValue + period - delta
                  : baseValue - delta);
                queue >>= bits;
                remaining += bits;
                if (fillsTimeline)
                {
                  WriteTimelineZero(h);
                }
                if (dataPos >= dataEnd)
                {
                  break;
                }
              }
              dataPos = packedEnd;
            }
          }

          // Финальный проход: масштабирование отсчётов и выравнивание шкалы времени
          for (h = currentNum; h < Timeline.Count; h++)
          {
            if (fillsTimeline)
            {
              Timeline[h] = Timeline[h - 1] + samplingStepUs;
            }
            int index = h - HeadLength;
            if (index < channel.Samples.Count)
            {
              channel.Samples[index] *= channel.K2 * channel.K7;
            }
            SearchMaxMin(channel, h);
          }
        }
        else
        {
          throw new InvalidOperationException($"Неизвестный метод сжатия канала ПАРМА: {channel.Method}.");
        }
      }

      /// <summary>
      /// Запись отсчёта по индексу точки (перезапись или добавление).
      /// Повторяет семантику oscils[d][h] из «Транскопа»: продолжение части
      /// затирает «хвост» повторов предыдущей части, а не дописывается после него.
      /// </summary>
      private static void WriteSample(AnalogData channel, int pointIndex, double value)
      {
        int index = pointIndex - HeadLength;
        if (index < channel.Samples.Count)
        {
          channel.Samples[index] = value;
        }
        else
        {
          channel.Samples.Add(value);
        }
      }

      /// <summary>
      /// Запись нуля в шкалу времени по индексу точки (cnt[h]=0 из «Транскопа»):
      /// существующий элемент перезаписывается, а присвоение за пределами массива
      /// удлиняет его до этого индекса. После «повтора окна» индекс h скачет на
      /// window вперёд — без заполнения промежутка шкала укоротится на 36 точек
      /// за каждый такой повтор (в JS промежуточные слоты — дырки, которые
      /// перезаписывает финальный проход).
      /// </summary>
      private void WriteTimelineZero(int pointIndex)
      {
        while (Timeline.Count <= pointIndex)
        {
          Timeline.Add(0);
        }
        Timeline[pointIndex] = 0;
      }

      private double SampleAt(AnalogData channel, int pointIndex)
      {
        int index = pointIndex - HeadLength;
        return index >= 0 && index < channel.Samples.Count ? channel.Samples[index] : 0;
      }

      /// <summary>Поиск локальных пиков (codecs.searchMaxMin) для оценки амплитуды канала.</summary>
      private static void SearchMaxMin(AnalogData channel, int pointIndex)
      {
        if (pointIndex <= HeadLength + 2)
        {
          return;
        }
        int current = pointIndex - HeadLength;
        if (current < 2)
        {
          return;
        }
        double abs = Math.Abs(channel.Samples[current]);
        double prev = Math.Abs(channel.Samples[current - 1]);
        if (prev > Math.Abs(channel.Samples[current - 2]) && abs <= prev)
        {
          if (prev > channel.MaxPeak)
          {
            channel.MaxPeak = prev;
          }
          if (prev < channel.MinPeak)
          {
            channel.MinPeak = prev;
          }
        }
      }

      /// <summary>Сдвиг частотных каналов к нулю (transcop.doFreqSpecial).</summary>
      private void AdjustFrequencyChannels()
      {
        int pointsPerPeriod = (int)Math.Floor(Sampling / BaseFrequency + 0.5);
        if (pointsPerPeriod == 0)
        {
          pointsPerPeriod = 10;
        }
        double periodSum = BaseFrequency * pointsPerPeriod;
        int limit = pointsPerPeriod + HeadLength;
        foreach (AnalogData channel in Analogs)
        {
          bool isHertz = Regex.IsMatch(channel.Unit, "^hz$", RegexOptions.IgnoreCase) || channel.Unit == "Гц";
          if (channel.ZeroOffset != 0 || !isHertz)
          {
            continue;
          }
          double sum = 0;
          for (int i = HeadLength; i < Math.Min(limit, HeadLength + channel.Samples.Count); i++)
          {
            sum += channel.Samples[i - HeadLength];
          }
          double shift = BaseFrequency * Math.Floor(sum / periodSum + 0.5);
          channel.ZeroOffset = -shift;
          channel.Nominal = Math.Abs(channel.MaxPeak - channel.MinPeak - shift) / 10;
          channel.MaxAmplitude = channel.MaxPeak - shift;
        }
      }

      /// <summary>Имя канала: всё до ':' — присоединение (ccbm), далее имя сигнала.</summary>
      private static void ReadChannelName(byte[] data, int pos, Encoding encoding, out string name, out string circuitComponent)
      {
        var full = new StringBuilder();
        var prefix = new StringBuilder();
        bool separatorFound = false;
        for (int i = 0; i < 66; i++)
        {
          int b = data[pos + i];
          if (b == 0)
          {
            break;
          }
          if (b == 58)
          {
            separatorFound = true;
          }
          string ch = encoding.GetString(data, pos + i, 1);
          full.Append(ch);
          if (!separatorFound)
          {
            prefix.Append(ch);
          }
        }
        name = full.ToString();
        circuitComponent = prefix.ToString();
      }

      /// <summary>Единица измерения (8 байт) с транскоповской логикой выделения содержимого скобок.</summary>
      private static string ReadUnit(byte[] data, int pos, Encoding encoding)
      {
        var unit = new StringBuilder();
        bool stopped = false;
        for (int i = 0; i < 8; i++, pos++)
        {
          int b = data[pos];
          if (b == 0)
          {
            stopped = true;
          }
          if (stopped || b == 41)
          {
            continue;
          }
          if (b == 40)
          {
            unit.Clear();
          }
          unit.Append(encoding.GetString(data, pos, 1));
        }
        return unit.ToString();
      }

      private static string ReadAsciiz(byte[] data, int from, int to, Encoding encoding)
      {
        int end = from;
        while (end < to && data[end] != 0)
        {
          end++;
        }
        return encoding.GetString(data, from, end - from);
      }

      /// <summary>Время из пары uint32 секунды + микросекунды, упакованное в одно число.</summary>
      private static double ReadPackedTime(byte[] data, int secondsPos)
      {
        double seconds = U32(data, secondsPos);
        double micro = U32(data, secondsPos + 4);
        if (micro >= 1000000)
        {
          seconds += 1;
          micro -= 1000000;
        }
        return seconds * 1000000.0 + micro;
      }

      /// <summary>Очередное расширение части записи: do -> d01 -> d02 ... (младший символ в base 36).</summary>
      private static string NextPartExtension(string current)
      {
        if (current == "do")
        {
          return "d01";
        }
        int lowIndex = mNumericSuffixChars.IndexOf(current[current.Length - 1]) + 1;
        int highIndex = mNumericSuffixChars.IndexOf(current[current.Length - 2]);
        if (lowIndex >= mNumericSuffixChars.Length)
        {
          lowIndex = 0;
          highIndex++;
        }
        return current.Substring(0, current.Length - 2) + mNumericSuffixChars[highIndex] + mNumericSuffixChars[lowIndex];
      }

      /// <summary>Фаза по имени канала (transcop.doPhase, поиск по UI-паттернам).</summary>
      private static string DetectPhase(string name)
      {
        for (int group = 0; group < mPhaseNamePatterns.Length; group++)
        {
          foreach (string pattern in mPhaseNamePatterns[group])
          {
            if (Regex.IsMatch(name, "(?:\\b|_)" + pattern + "\\b", RegexOptions.IgnoreCase))
            {
              return new[] { "AB", "BC", "CA", "A", "B", "C", "N" }[group];
            }
          }
        }
        return "";
      }
    }

    // ===== Модель записи =============================================================

    private static OscillogramData CreateRecord(ParmaRecord record)
    {
      int pointCount = record.Timeline.Count - HeadLength;
      if (pointCount <= 0)
      {
        throw new InvalidOperationException("Осциллограмма ПАРМА не содержит точек данных.");
      }

      var analogChannels = new List<AnalogChannel>();
      var divisors = new List<double>();
      foreach (AnalogData channel in record.Analogs)
      {
        // a = (|макс|+|мин|)/5000 с восемью значащими цифрами, как в save.createCFG.
        double scale = ToPrecision8((Math.Abs(channel.MaxPeak) + Math.Abs(channel.MinPeak)) / 5000.0);
        divisors.Add(1.0 / scale);
        double range = Math.Ceiling(Math.Abs(channel.MaxAmplitude));
        analogChannels.Add(new AnalogChannel(
          number: analogChannels.Count + 1,
          name: channel.Name,
          phase: channel.Phase,
          circuitComponent: channel.CircuitComponent,
          unit: channel.Unit,
          a: scale,
          b: channel.ZeroOffset,
          skew: 0,
          min: -range,
          max: range,
          primary: channel.K1,
          secondary: 1,
          signalType: SignalType.Secondary));
      }

      var discreteChannels = new List<DiscreteChannel>();
      foreach (DiscreteData channel in record.Discretes)
      {
        discreteChannels.Add(new DiscreteChannel(
          number: discreteChannels.Count + 1,
          name: channel.Name,
          phase: "",
          circuitComponent: channel.CircuitComponent,
          normalState: false));
      }

      double[][] analogColumns = new double[analogChannels.Count][];
      for (int j = 0; j < analogChannels.Count; j++)
      {
        double[] col = analogColumns[j] = new double[pointCount];
        var samples = record.Analogs[j].Samples;
        double div = divisors[j];
        int count = Math.Min(pointCount, samples.Count);
        for (int i = 0; i < count; i++)
        {
          col[i] = samples[i] * div;
        }
      }

      bool[][] discreteColumns = new bool[discreteChannels.Count][];
      for (int j = 0; j < discreteChannels.Count; j++)
      {
        bool[] col = discreteColumns[j] = new bool[pointCount];
        var disc = record.Discretes[j];
        var events = disc.EventTimes;
        int eventIdx = 1;
        int curVal = disc.InitialState;
        for (int i = 0; i < pointCount; i++)
        {
          double timeUs = record.Timeline[HeadLength + i];
          while (eventIdx < events.Count && events[eventIdx] < timeUs)
          {
            curVal ^= 1;
            eventIdx++;
          }
          col[i] = curVal > 0;
        }
      }

      long[] timestampsUs = new long[pointCount];
      for (int i = 0; i < pointCount; i++)
      {
        timestampsUs[i] = (long)Math.Floor(record.Timeline[HeadLength + i] + 0.5);
      }

      var configuration = new ConfigurationHandler
      {
        StationName = "",
        DeviceId = (record.InstrumentName ?? "").Split(',')[0],
        Version = ComtradeVersion.V2013,
        AnalogChannels = analogChannels,
        DiscreteChannels = discreteChannels,
        SampleRates = new List<SampleRate> { new SampleRate(record.Sampling, pointCount) },
        Frequency = BaseFrequency,
        StartTime = FromUnixMicroseconds(record.BeginTimeUs),
        TriggerTime = FromUnixMicroseconds(record.TriggerTimeUs),
        DataFileType = DataFileType.ASCII,
        TimeMultiplicationFactor = 1
      };

      return NativeSampleBuilder.Create(configuration, analogColumns, discreteColumns, timestampsUs);
    }

    private static DateTime FromUnixMicroseconds(double microseconds)
    {
      var epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
      return epoch.AddTicks((long)Math.Floor(microseconds * 10.0 + 0.5));
    }

    private static double ToPrecision8(double value) => BinaryDecimal.Round(value, 8);
  }
}
