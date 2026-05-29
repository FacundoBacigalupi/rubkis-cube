using System.Collections.Generic;

public class RubikMoveHistory
{
    public static RubikMoveHistory Instance { get; } = new RubikMoveHistory();

    private readonly Stack<RubikMove> history = new Stack<RubikMove>();

    public int Count => history.Count;

    public void Push(RubikMove move) => history.Push(move);

    public RubikMove Pop() => history.Pop();

    public bool IsEmpty => history.Count == 0;

    public void Clear() => history.Clear();
}
