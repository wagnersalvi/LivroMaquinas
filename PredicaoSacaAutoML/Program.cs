using Microsoft.ML;
using Microsoft.ML.AutoML;

namespace PredicaoSacaAutoML
{


	class Program
	{
		static async Task Main(string[] args)
		{
			MLContext contextoML = new MLContext();
			//contextoML.Log += (sender, e) => Console.WriteLine(e.Message);

			// Importar ou criar dados de treinamento
			var rand = new Random();
			var dadosCasas = new List<DadosImovel>();

			for (int i = 0; i < 100; i++)
			{
				// Gerar um tamanho entre 50 e 500m²
				float tamanho = (float)(rand.NextDouble() * 450 + 50);
				// Definir preço como uma função do tamanho com uma variação aleatória
				// Aqui, preço é aproximadamente tamanho * 0.8 com uma variação aleatória
				float preco = tamanho * 0.8f + (float)(rand.NextDouble() * 10 - 5);

				dadosCasas.Add(new DadosImovel { Tamanho = tamanho, Preco = preco });
			}

			// Imprimir os dados gerados (opcional)
			foreach (var dados in dadosCasas)
			{
				Console.WriteLine($"Tamanho: {dados.Tamanho:F1}, Preço: {dados.Preco:F1}");
			}
			IDataView dadosTreinamento = contextoML.Data.LoadFromEnumerable(dadosCasas);

			// Definir tarefa de regressão
			var experiment = contextoML.Auto().CreateRegressionExperiment(maxExperimentTimeInSeconds: 60);

			// Configurar treinamento
			var result = experiment.Execute(dadosTreinamento, labelColumnName: "Preco");

			// Avaliar o modelo
			Console.WriteLine($"Melhor modelo: {result.BestRun.TrainerName}");
			Console.WriteLine($"R-squared: {result.BestRun.ValidationMetrics.RSquared}");

			// Fazer predições
			var model = result.BestRun.Model;
			var input = new DadosImovel { Tamanho = 150.0F };
			var predictionFunction = contextoML.Model.CreatePredictionEngine<DadosImovel, PredicaoImovel>(model);
			var prediction = predictionFunction.Predict(input);

			Console.WriteLine($"Preço previsto: {prediction.PrecoPrevisao}");
		}
	}
}