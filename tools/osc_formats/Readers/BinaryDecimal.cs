namespace OscFormats.Native
{
  using System;
  using System.Globalization;
  using System.Numerics;

  internal static class BinaryDecimal
  {
    /// <summary>Округление точного двоичного значения до заданного числа значащих цифр</summary>
    public static double Round(double value, int digits)
    {
      if (value == 0)
      {
        return 0;
      }
      // Округляем точное двоичное значение, без предварительного округления до 15 цифр
      // G в .NET Framework сначала округляет до 15 цифр и меняет пограничные случаи
      long bits = BitConverter.DoubleToInt64Bits(Math.Abs(value));
      int exponent = (int)((bits >> 52) & 0x7ff);
      long mantissa = bits & 0xfffffffffffff;
      if (exponent != 0) { mantissa |= 1L << 52; }
      int binaryPower = exponent == 0 ? -1074 : exponent - 1075;
      BigInteger numerator = mantissa;
      BigInteger denominator = BigInteger.One;
      if (binaryPower >= 0) { numerator <<= binaryPower; }
      else { denominator <<= -binaryPower; }
      int decimalPower = digits - 1 - (int)Math.Floor(Math.Log10(Math.Abs(value)));
      if (decimalPower >= 0) { numerator *= BigInteger.Pow(10, decimalPower); }
      else { denominator *= BigInteger.Pow(10, -decimalPower); }
      var rounded = BigInteger.DivRem(numerator, denominator, out BigInteger remainder);
      if (2 * remainder >= denominator) { rounded++; }
      string text = (value < 0 ? "-" : "") + rounded.ToString(CultureInfo.InvariantCulture) + "E" + (-decimalPower).ToString(CultureInfo.InvariantCulture);
      return double.Parse(text, CultureInfo.InvariantCulture);
    }
  }
}
