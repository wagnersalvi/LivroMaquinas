using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;

class Program
{
	static void Main()
	{
		// Inicializa o contexto ML.NET
		var mlContext = new MLContext();

		// Dados históricos de vendas (exemplo: vendas semanais)
		var dadosVendas = new[]
		{
			new Venda() { Semana = 1, Quantidade = 200 },
			new Venda() { Semana = 2, Quantidade = 250 },
			new Venda() { Semana = 3, Quantidade = 270 },
			new Venda() { Semana = 4, Quantidade = 300 },
			new Venda() { Semana = 5, Quantidade = 320 },
			new Venda() { Semana = 6, Quantidade = 340 },
			new Venda() { Semana = 7, Quantidade = 390 },
			new Venda() { Semana = 8, Quantidade = 410 }
		};

		// Converte os dados para IDataView
		var dataView = mlContext.Data.LoadFromEnumerable(dadosVendas);

		// Configura a pipeline para previsão de séries temporais
		var pipeline = mlContext.Forecasting.ForecastBySsa(
			outputColumnName: nameof(PrevisaoVendas.QuantidadePrevista),
			inputColumnName: nameof(Venda.Quantidade),
			windowSize: 3,       // Número de pontos usados para análise
			seriesLength: 8,      // Tamanho da série completa
			trainSize: 8,         // Quantidade de dados usados para treinar
			horizon: 3);          // Número de previsões futuras

		// Treina o modelo
		var model = pipeline.Fit(dataView);

		// Criando um mecanismo de previsão
		var forecastingEngine = model.CreateTimeSeriesEngine<Venda, PrevisaoVendas>(mlContext);

		// Fazendo previsões para as próximas 3 semanas
		var previsao = forecastingEngine.Predict();

		// Exibe os resultados
		Console.WriteLine("Previsão de vendas para as próximas semanas:");
		for (int i = 0; i < previsao.QuantidadePrevista.Length; i++)
		{
			Console.WriteLine($"Semana {dadosVendas.Length + i + 1}: {previsao.QuantidadePrevista[i]:0}");
		}


	}
}

