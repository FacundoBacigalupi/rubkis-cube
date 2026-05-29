public readonly struct RubikMove
{
    public readonly RubikAxis Axis;
    public readonly int Layer;     // -1, 0, 1
    public readonly int Direction; // -1 or +1

    public RubikMove(RubikAxis axis, int layer, int direction)
    {
        Axis = axis;
        Layer = layer;
        Direction = direction >= 0 ? 1 : -1;
    }

    public RubikMove Inverse()
    {
        return new RubikMove(Axis, Layer, -Direction);
    }

    public override string ToString()
    {
        string dir = Direction > 0 ? "" : "'";
        return $"{Axis}{Layer}{dir}";
    }
}
