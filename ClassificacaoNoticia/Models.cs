using Microsoft.ML.Data;

namespace ClassificacaoNoticia;

public class Noticia
{
	public string TextoArtigo { get; set; }
	public string Categoria { get; set; }
}

public class PredicaoNoticia
{
	[ColumnName("PredictedLabel")]
	public string Categoria { get; set; }

	public float[] Score { get; set; } 
}