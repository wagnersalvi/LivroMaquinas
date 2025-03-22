using Microsoft.ML.Data;

namespace RegressaoLinear;

public class DadosImovel
{
	[LoadColumn(0)]
	public float Tamanho { get; set; }

	[LoadColumn(1)]
	public float Quartos { get; set; }

	[LoadColumn(2)]
	public float Preco { get; set; }
}

public class PredicaoImovel
{
	[ColumnName("Score")]
	public float PrecoPrevisao { get; set; }
}