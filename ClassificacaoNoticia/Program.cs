using ClassificacaoNoticia;
using Microsoft.ML;
using Microsoft.ML.Data;



class Program
{
	static void Main(string[] args)
	{
		var mlContext = new MLContext();

		// Dados de treinamento
		var noticias = new List<Noticia>
		{
			new Noticia { TextoArtigo = "O uso de inteligência artificial...", Categoria = "Tecnologia" },
			new Noticia { TextoArtigo = "Computadores avançados para empresas", Categoria = "Tecnologia" },
			new Noticia { TextoArtigo = "Final do campeonato mundial de futebol", Categoria = "Esportes" },
			new Noticia { TextoArtigo = "Novos recordes na NBA", Categoria = "Esportes" },
			new Noticia { TextoArtigo = "Eleições presidenciais em debate", Categoria = "Política" },
			new Noticia { TextoArtigo = "Reforma tributária em discussão", Categoria = "Política" }
		};

		// Carregar os dados no ML.NET
		var trainData = mlContext.Data.LoadFromEnumerable(noticias);

		// Pipeline de treinamento
		var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(Noticia.Categoria)) // Converte categoria para chave numérica
			.Append(mlContext.Transforms.Text.FeaturizeText("Features", nameof(Noticia.TextoArtigo))) // Vetorização do texto
			.Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features")) // Modelo de classificação
			.Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "Label")); // Converte chave numérica de volta para string

		// Treinar o modelo
		var model = pipeline.Fit(trainData);

		// Criar um motor de predição
		var predictionEngine = mlContext.Model.CreatePredictionEngine<Noticia, PredicaoNoticia>(model);

		// Testar uma nova notícia
		var newArticle = new Noticia { TextoArtigo = "Computador do presidente explodiu assistindo futebol" };
		var prediction = predictionEngine.Predict(newArticle);

		// Obter os nomes das categorias manualmente
		var categorias = noticias.Select(n => n.Categoria).Distinct().OrderBy(c => c).ToList();

		var predictions = model.Transform(trainData);

		// Exibir a predição
		Console.WriteLine($"Artigo: {newArticle.TextoArtigo}");

		// Diagnóstico: verificar se há valores
		if (!string.IsNullOrEmpty(prediction.Categoria))
		{
			Console.WriteLine($"Categoria Predita: {prediction.Categoria}\n");
		}
		else
		{
			int categoriaIndex = Array.IndexOf(prediction.Score, prediction.Score.Max());
			string categoriaPredita = categorias[categoriaIndex];

			Console.WriteLine($"\nCategoria Predita: {categoriaPredita}");

		}

		// Exibir pontuação das categorias
		Console.WriteLine("Pontuação das categorias:");
		for (int i = 0; i < prediction.Score.Length; i++)
		{
			string categoriaNome = (i < categorias.Count) ? categorias[i] : $"Categoria_{i}";
			Console.WriteLine($"{categoriaNome}: {prediction.Score[i]:F4}");
		}

		// ****************** AVALIAÇÃO DO MODELO ******************
		Console.WriteLine("\nAvaliando o modelo...");

		// Criar conjunto de teste
		var testNoticias = new List<Noticia>
		{
			new Noticia { TextoArtigo = "Nova tecnologia de smartphones...", Categoria = "Tecnologia" },
			new Noticia { TextoArtigo = "Time de basquete vence o campeonato", Categoria = "Esportes" },
			new Noticia { TextoArtigo = "Eleições para governador acontecem neste domingo", Categoria = "Política" }
		};

		var testData = mlContext.Data.LoadFromEnumerable(testNoticias);

		// Fazer previsões no conjunto de teste
		var transformedData = model.Transform(testData);

		// Avaliação do modelo
		var metrics = mlContext.MulticlassClassification.Evaluate(transformedData, labelColumnName: "Label", predictedLabelColumnName: "PredictedLabel");

		// Exibir métricas do modelo
		Console.WriteLine($"\nMétricas do Modelo:");
		Console.WriteLine($"Acurácia Macro: {metrics.MacroAccuracy:P2}");
		Console.WriteLine($"Acurácia Micro: {metrics.MicroAccuracy:P2}");
		Console.WriteLine($"Log-Loss: {metrics.LogLoss:F4}");
		Console.WriteLine($"Log-Loss por Classe: {string.Join(", ", metrics.PerClassLogLoss.Select(l => l.ToString("F4")))}");

		Console.WriteLine("\nAvaliação concluída!");
	}
}
