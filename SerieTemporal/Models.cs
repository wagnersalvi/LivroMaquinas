// Classe representando os dados de entrada
using Microsoft.ML.Data;

public class Venda
{
	[LoadColumn(0)]
	public float Semana { get; set; }

	[LoadColumn(1)]
	public float Quantidade { get; set; }
}

// Classe para armazenar os resultados da previsão
public class PrevisaoVendas
{
	[ColumnName("QuantidadePrevista")]
	public float[] QuantidadePrevista { get; set; }
}


public class Anomalia
{
	public float Semana { get; set; }

	[ColumnName("Alert")]
	public float PontuacaoAnomalia { get; set; }
}