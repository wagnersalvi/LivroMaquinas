using Microsoft.ML.Data;

namespace RegressaoLogistica;

public class DadosSentimento
{
	[LoadColumn(0)]
	public string FraseSentimento { get; set; }

	[LoadColumn(1), ColumnName("Label")]
	public bool SentimentoBom { get; set; }
}

// Classe para representar a previsão
public class PredicaoSentimento
{
	[ColumnName("PredictedLabel")]
	public bool Predicao { get; set; }

	public float Probabilidade { get; set; }

	public float Score { get; set; }
}
