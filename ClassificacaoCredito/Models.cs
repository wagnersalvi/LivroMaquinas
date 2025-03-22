using Microsoft.ML.Data;

namespace ClassificacaoCredito;

public class DadosCredito
{
	public float Renda { get; set; }
	public float Dividas { get; set; }
	public bool Aprovado { get; set; }
}

public class PredicaoCredito
{
	[ColumnName("PredictedLabel")]
	public bool PredictedLabel { get; set; }

	[ColumnName("Probability")]
	public float Probability { get; set; }

	[ColumnName("Score")]
	public float Score { get; set; }
}