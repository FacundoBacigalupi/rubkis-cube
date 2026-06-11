using System;

/// <summary>
/// Pure C# Rubik's Cube simulator on a flat byte[54] array.
/// Mirrors rubikml/cube.py exactly: same face order, same move logic, same row/col conventions.
///
/// Layout:  index = face * 9 + row * 3 + col
/// Faces:   U=0  D=1  R=2  L=3  F=4  B=5
/// Colors:  same integer as the face the sticker belongs to in the solved state
/// </summary>
public sealed class RubikCubeArray
{
    const int U = 0, D = 1, R = 2, L = 3, F = 4, B = 5;

    public readonly byte[] S = new byte[54];

    static readonly byte[] SolvedTemplate =
    {
        0,0,0, 0,0,0, 0,0,0,  // U  (indices  0- 8)
        1,1,1, 1,1,1, 1,1,1,  // D  (indices  9-17)
        2,2,2, 2,2,2, 2,2,2,  // R  (indices 18-26)
        3,3,3, 3,3,3, 3,3,3,  // L  (indices 27-35)
        4,4,4, 4,4,4, 4,4,4,  // F  (indices 36-44)
        5,5,5, 5,5,5, 5,5,5,  // B  (indices 45-53)
    };

    public static RubikCubeArray Solved()
    {
        var c = new RubikCubeArray();
        Array.Copy(SolvedTemplate, c.S, 54);
        return c;
    }

    public RubikCubeArray Clone()
    {
        var c = new RubikCubeArray();
        Array.Copy(S, c.S, 54);
        return c;
    }

    public bool IsSolved()
    {
        for (int i = 0; i < 54; i++)
            if (S[i] != SolvedTemplate[i]) return false;
        return true;
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    static int _f(int face, int row, int col) => face * 9 + row * 3 + col;

    void RotateFaceCw(int face)
    {
        int f = face * 9;
        byte t0 = S[f+0], t1 = S[f+1], t2 = S[f+2],
             t3 = S[f+3], t4 = S[f+4], t5 = S[f+5],
             t6 = S[f+6], t7 = S[f+7], t8 = S[f+8];
        // CW permutation: position (r,c) ← (2-c, r)
        S[f+0]=t6; S[f+1]=t3; S[f+2]=t0;
        S[f+3]=t7; S[f+4]=t4; S[f+5]=t1;
        S[f+6]=t8; S[f+7]=t5; S[f+8]=t2;
    }

    // ── forward moves (exact ports of Python's _R, _L, … functions) ──────────

    void MoveR()
    {
        RotateFaceCw(R);
        for (int i = 0; i < 3; i++)
        {
            byte tmp    = S[_f(U, i,   2)];
            S[_f(U, i,   2)] = S[_f(F, i,   2)];
            S[_f(F, i,   2)] = S[_f(D, i,   2)];
            S[_f(D, i,   2)] = S[_f(B, 2-i, 0)];
            S[_f(B, 2-i, 0)] = tmp;
        }
    }

    void MoveL()
    {
        RotateFaceCw(L);
        for (int i = 0; i < 3; i++)
        {
            byte tmp       = S[_f(U, i,   0)];
            S[_f(U, i,   0)] = S[_f(B, 2-i, 2)];
            S[_f(B, 2-i, 2)] = S[_f(D, i,   0)];
            S[_f(D, i,   0)] = S[_f(F, i,   0)];
            S[_f(F, i,   0)] = tmp;
        }
    }

    void MoveU()
    {
        RotateFaceCw(U);
        for (int i = 0; i < 3; i++)
        {
            byte tmp    = S[_f(F, 0, i)];
            S[_f(F, 0, i)] = S[_f(R, 0, i)];
            S[_f(R, 0, i)] = S[_f(B, 0, i)];
            S[_f(B, 0, i)] = S[_f(L, 0, i)];
            S[_f(L, 0, i)] = tmp;
        }
    }

    void MoveD()
    {
        RotateFaceCw(D);
        for (int i = 0; i < 3; i++)
        {
            byte tmp    = S[_f(F, 2, i)];
            S[_f(F, 2, i)] = S[_f(L, 2, i)];
            S[_f(L, 2, i)] = S[_f(B, 2, i)];
            S[_f(B, 2, i)] = S[_f(R, 2, i)];
            S[_f(R, 2, i)] = tmp;
        }
    }

    void MoveF()
    {
        RotateFaceCw(F);
        for (int i = 0; i < 3; i++)
        {
            byte tmp         = S[_f(U, 2,   i)];
            S[_f(U, 2,   i)] = S[_f(L, 2-i, 2)];
            S[_f(L, 2-i, 2)] = S[_f(D, 0,   2-i)];
            S[_f(D, 0, 2-i)] = S[_f(R, i,   0)];
            S[_f(R, i,   0)] = tmp;
        }
    }

    void MoveB()
    {
        RotateFaceCw(B);
        for (int i = 0; i < 3; i++)
        {
            byte tmp           = S[_f(U, 0,   i)];
            S[_f(U, 0,   i)]   = S[_f(R, i,   2)];
            S[_f(R, i,   2)]   = S[_f(D, 2,   2-i)];
            S[_f(D, 2, 2-i)]   = S[_f(L, 2-i, 0)];
            S[_f(L, 2-i, 0)]   = tmp;
        }
    }

    // ── public API ───────────────────────────────────────────────────────────

    /// <summary>
    /// Apply move by index.  Order matches Python MOVES list exactly:
    /// 0=R  1=R'  2=L  3=L'  4=U  5=U'  6=D  7=D'  8=F  9=F'  10=B  11=B'
    /// </summary>
    public void ApplyMove(int idx)
    {
        switch (idx)
        {
            case  0: MoveR(); break;
            case  1: MoveR(); MoveR(); MoveR(); break;  // R' = R×3
            case  2: MoveL(); break;
            case  3: MoveL(); MoveL(); MoveL(); break;
            case  4: MoveU(); break;
            case  5: MoveU(); MoveU(); MoveU(); break;
            case  6: MoveD(); break;
            case  7: MoveD(); MoveD(); MoveD(); break;
            case  8: MoveF(); break;
            case  9: MoveF(); MoveF(); MoveF(); break;
            case 10: MoveB(); break;
            case 11: MoveB(); MoveB(); MoveB(); break;
        }
    }

    /// <summary>
    /// Fill a pre-allocated float[324] with the one-hot encoding used by the ONNX model.
    /// Sticker i with color c → output[i*6 + c] = 1.
    /// </summary>
    public void EncodeOneHot(float[] output)
    {
        Array.Clear(output, 0, 324);
        for (int i = 0; i < 54; i++)
            output[i * 6 + S[i]] = 1f;
    }

    /// <summary>
    /// Write one-hot encoding into a sub-region of a larger batch buffer starting at offset.
    /// Used for batched inference: batch[n] starts at n*324.
    /// </summary>
    public void EncodeOneHot(float[] batchBuffer, int offset)
    {
        Array.Clear(batchBuffer, offset, 324);
        for (int i = 0; i < 54; i++)
            batchBuffer[offset + i * 6 + S[i]] = 1f;
    }

    /// <summary>
    /// Reconstruct state from a float[324] one-hot buffer (written by CubeStateReader).
    /// </summary>
    public static RubikCubeArray FromOneHot(float[] input)
    {
        var c = new RubikCubeArray();
        for (int i = 0; i < 54; i++)
            for (int col = 0; col < 6; col++)
                if (input[i * 6 + col] > 0.5f) { c.S[i] = (byte)col; break; }
        return c;
    }
}
