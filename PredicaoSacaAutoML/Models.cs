using Microsoft.ML.Data;

namespace PredicaoSacaAutoML;

public class DadosImovel
{
	public float Tamanho { get; set; }
	public float Preco { get; set; }
}

public class PredicaoImovel
{
	[ColumnName("Score")]
	public float PrecoPrevisao { get; set; }
}