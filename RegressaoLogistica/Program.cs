using Microsoft.ML;

namespace RegressaoLogistica;


class Program
{
	static void Main(string[] args)
	{
		// Criar contexto ML
		MLContext mlContext = new MLContext(seed: 0);

		// Preparar dados
		var sentimentos = new List<DadosSentimento>
		{
			new DadosSentimento { FraseSentimento = "Eu adoro este produto!", SentimentoBom = true },
			new DadosSentimento { FraseSentimento = "Produto horrivel", SentimentoBom = false },
			new DadosSentimento { FraseSentimento = "Ótima surpresa", SentimentoBom = true },
			new DadosSentimento { FraseSentimento = "Nunca mais compro este produto", SentimentoBom = false }
		};
		IDataView dadosTreinamento = mlContext.Data.LoadFromEnumerable(sentimentos);

		// Definir pipeline de treinamento

		var pipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "FraseSentimento")
			.Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

		// Treinar o modelo
		var model = pipeline.Fit(dadosTreinamento);

		// Criar motor de previsão
		var predictionEngine = mlContext.Model.CreatePredictionEngine<DadosSentimento, PredicaoSentimento>(model);

		// Fazer previsões
		var testSentiment = new DadosSentimento { FraseSentimento = "Eu estou feliz com esta compra" };
		var prediction = predictionEngine.Predict(testSentiment);

		Console.WriteLine($"Sentimento: {testSentiment.FraseSentimento}");
		Console.WriteLine($"Previsão: {(prediction.Predicao ? "Positivo" : "Negativo")}");
		Console.WriteLine($"Probabilidade: {prediction.Probabilidade:P2}");


		testSentiment = new DadosSentimento { FraseSentimento = "Produto lixo" };
		prediction = predictionEngine.Predict(testSentiment);

		Console.WriteLine($"Sentimento: {testSentiment.FraseSentimento}");
		Console.WriteLine($"Previsão: {(prediction.Predicao ? "Positivo" : "Negativo")}");
		Console.WriteLine($"Probabilidade: {prediction.Probabilidade:P2}");

	}
}