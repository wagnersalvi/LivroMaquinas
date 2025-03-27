using Microsoft.ML.Data;

namespace ExemploNLP;

public class TreinamentoSentimento
{
	public string Opiniao { get; set; }
	public string Avaliacao { get; set; }
}

public class PredicaoSentimento
{
	[ColumnName("PredictedLabel")]
	public string Avaliacao { get; set; }

	public float[] Score { get; set; }
}