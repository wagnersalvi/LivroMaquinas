using ClassificacaoCredito;
using Microsoft.ML;

class Program
{
	static void Main(string[] args)
	{
		var mlContext = new MLContext();

		// Dados fictícios de análise de crédito
		var trainData = new List<DadosCredito>
		{
			new DadosCredito { Renda = 50000, Dividas = 20000, Aprovado = true },
			new DadosCredito { Renda = 40000, Dividas = 30000, Aprovado = false },
			new DadosCredito { Renda = 75000, Dividas = 10000, Aprovado = true },
			new DadosCredito { Renda = 30000, Dividas = 40000, Aprovado = false }
		};

		var trainingData = mlContext.Data.LoadFromEnumerable(trainData);

		// Pipeline de FastTree
		var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "Renda", "Dividas" })
			.Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Aprovado", featureColumnName: "Features"));

		//Pipeline de SdcaLogisticRegression
		//var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "Renda", "Dividas" })
		//	   .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Aprovado", featureColumnName: "Features"));

		// Treinando o modelo
		var model = pipeline.Fit(trainingData);

		// Fazendo previsões
		var predictionEngine = mlContext.Model.CreatePredictionEngine<DadosCredito, PredicaoCredito>(model);

		var newCreditData = new DadosCredito { Renda = 60000, Dividas = 15000 };
		var prediction = predictionEngine.Predict(newCreditData);

		Console.WriteLine($"Renda: {newCreditData.Renda}, Dívida: {newCreditData.Dividas}");
		Console.WriteLine($"Aprovação do Empréstimo? {(prediction.PredictedLabel ? "Sim" : "Não")}");
		Console.WriteLine($"Probabilidade: {prediction.Probability:P2}");
	}
}