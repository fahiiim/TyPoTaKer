using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

namespace TyPoTaKer.Desktop.Engine;

/// <summary>
/// Bit-for-bit compatible tokenizer matching HuggingFace ByteLevel BPE used by TyPoFormer.
/// </summary>
public sealed class TyPoTokenizer
{
    private static readonly Dictionary<byte, char> ByteToChar = new();
    private static readonly Dictionary<char, byte> CharToByte = new();
    private static readonly Regex Gpt2Regex = new(
        @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled
    );

    static TyPoTokenizer()
    {
        var bs = new List<int>();
        for (int i = '!'; i <= '~'; i++) bs.Add(i);
        for (int i = '¡'; i <= '¬'; i++) bs.Add(i);
        for (int i = '®'; i <= 'ÿ'; i++) bs.Add(i);

        var cs = new List<int>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!bs.Contains(b))
            {
                bs.Add(b);
                cs.Add(256 + n);
                n++;
            }
        }
        for (int i = 0; i < bs.Count; i++)
        {
            ByteToChar[(byte)bs[i]] = (char)cs[i];
            CharToByte[(char)cs[i]] = (byte)bs[i];
        }
    }

    private readonly BpeTokenizer _bpe;

    public int PadId => 0;
    public int UnkId => 1;
    public int BosId => 2;
    public int EosId => 3;
    public int MaxCharsPerToken { get; }

    public TyPoTokenizer(string vocabPath, string mergesPath, int maxCharsPerToken = 16)
    {
        if (!File.Exists(vocabPath))
            throw new FileNotFoundException($"Vocab file not found: {vocabPath}");
        if (!File.Exists(mergesPath))
            throw new FileNotFoundException($"Merges file not found: {mergesPath}");

        _bpe = BpeTokenizer.Create(vocabPath, mergesPath);
        MaxCharsPerToken = maxCharsPerToken;
    }

    public static string ConvertBytesToByteLevel(string input)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(input);
        var sb = new StringBuilder(bytes.Length);
        foreach (byte b in bytes)
        {
            sb.Append(ByteToChar[b]);
        }
        return sb.ToString();
    }

    public static string DecodeByteLevel(string byteLevelStr)
    {
        var bytes = new List<byte>(byteLevelStr.Length);
        foreach (char c in byteLevelStr)
        {
            if (CharToByte.TryGetValue(c, out byte b))
                bytes.Add(b);
        }
        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    public (List<long> TokenIds, List<string> TokenPieces) Encode(string text, bool addBos = true, bool addEos = false)
    {
        var tokenIds = new List<long>();
        var tokenPieces = new List<string>();

        if (addBos)
        {
            tokenIds.Add(BosId);
            tokenPieces.Add("<bos>");
        }

        var matches = Gpt2Regex.Matches(text);
        foreach (Match m in matches)
        {
            string byteStr = ConvertBytesToByteLevel(m.Value);
            var ids = _bpe.EncodeToIds(byteStr, considerPreTokenization: false, considerNormalization: false);
            foreach (int id in ids)
            {
                tokenIds.Add(id);
                tokenPieces.Add(_bpe.Decode(new List<int> { id }));
            }
        }

        if (addEos)
        {
            tokenIds.Add(EosId);
            tokenPieces.Add("<eos>");
        }

        return (tokenIds, tokenPieces);
    }

    public string Decode(IEnumerable<int> ids)
    {
        var filtered = ids.Where(id => id > 3).ToList();
        if (filtered.Count == 0) return string.Empty;
        string byteLevel = _bpe.Decode(filtered);
        return DecodeByteLevel(byteLevel);
    }

    public long[,,] BuildCharTensor(IReadOnlyList<string> tokenPieces)
    {
        int seqLen = tokenPieces.Count;
        var tensor = new long[1, seqLen, MaxCharsPerToken];

        for (int t = 0; t < seqLen; t++)
        {
            byte[] raw = Encoding.UTF8.GetBytes(tokenPieces[t]);
            for (int c = 0; c < MaxCharsPerToken; c++)
            {
                if (c < raw.Length)
                {
                    byte b = raw[c];
                    tensor[0, t, c] = (b > 0 && b < 128) ? b : 127;
                }
                else
                {
                    tensor[0, t, c] = 0;
                }
            }
        }

        return tensor;
    }
}
