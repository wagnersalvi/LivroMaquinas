using ExemploNLP;
using Microsoft.ML;

class Program
{
	static void Main(string[] args)
	{
		var mlContext = new MLContext();

		// Dados de treinamento
		var treinemento = new List<TreinamentoSentimento>
		{
			new TreinamentoSentimento { Opiniao = "Adorei o produto, atendendo minhas expectativas", Avaliacao = "Boa" },
			new TreinamentoSentimento { Opiniao = "Funcionamento perfeito", Avaliacao = "Boa" },
			new TreinamentoSentimento { Opiniao = "Aprovado e muito útil", Avaliacao = "Boa" },
			new TreinamentoSentimento { Opiniao = "Recomendo o produto", Avaliacao = "Boa" },
			new TreinamentoSentimento { Opiniao = "Bom investimento", Avaliacao = "Boa" },

			new TreinamentoSentimento { Opiniao = "Não vale o investimento", Avaliacao = "Ruim" },
			new TreinamentoSentimento { Opiniao = "Não funciona", Avaliacao = "Ruim" },
			new TreinamentoSentimento { Opiniao = "Produto parou de funcionar", Avaliacao = "Ruim" },
			new TreinamentoSentimento { Opiniao = "Devolvi por que não me atendeu", Avaliacao = "Ruim" },

		};

		// Carregar os dados no ML.NET
		var trainData = mlContext.Data.LoadFromEnumerable(treinemento);

		// Pipeline de treinamento
		var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(TreinamentoSentimento.Avaliacao)) // Converte avaliacao para chave numérica
			.Append(mlContext.Transforms.Text.FeaturizeText("Features", nameof(TreinamentoSentimento.Opiniao))) // Vetorização do texto
			.Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features")) // Modelo de classificação
			.Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "Label")); // Converte chave numérica de volta para string

		// Treinar o modelo
		var model = pipeline.Fit(trainData);

		// Criar um motor de predição
		var predictionEngine = mlContext.Model.CreatePredictionEngine<TreinamentoSentimento, PredicaoSentimento>(model);

		// Testar uma uma nova opinião
		var novaopiniao = new TreinamentoSentimento { Opiniao = "Atendeu minhas expectativas" };
		var prediction = predictionEngine.Predict(novaopiniao);

		// Obter os nomes das avaliacao manualmente
		var avaliacao = treinemento.Select(n => n.Avaliacao).Distinct().OrderBy(c => c).ToList();

		var predictions = model.Transform(trainData);

		// Exibir a predição
		Console.WriteLine($"Opinião: {novaopiniao.Opiniao}");

		// Diagnóstico: verificar se há valores
		if (!string.IsNullOrEmpty(prediction.Avaliacao))
		{
			Console.WriteLine($"Avaliacao Predita: {prediction.Avaliacao}\n");
		}
		else
		{
			int avaliacaoIndex = Array.IndexOf(prediction.Score, prediction.Score.Max());
			string avaliacaoPredita = avaliacao[avaliacaoIndex];

			Console.WriteLine($"\nAvaliação Predita: {avaliacaoPredita}");

		}

		// Exibir pontuação das categorias
		Console.WriteLine("Pontuação das Avaliações:");
		for (int i = 0; i < prediction.Score.Length; i++)
		{
			string avaliacaoNome = (i < avaliacao.Count) ? avaliacao[i] : $"Avaliacao_{i}";
			Console.WriteLine($"{avaliacaoNome}: {prediction.Score[i]:F4}");
		}

	}
}
