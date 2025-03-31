using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

public class Program
{
	public class ProductRating
	{
		[KeyType(10)]
		public uint UserId { get; set; }

		[KeyType(10)]
		public uint ProductId { get; set; }

		public float Rating { get; set; }
	}

	public class PredictionResult
	{
		public float Score;
	}

	static void Main()
	{
		var context = new MLContext();

		// Dados corrigidos e ampliados
		var trainingData = new[]
		{
			new ProductRating { UserId = 1u, ProductId = 101u, Rating = 5f },
			new ProductRating { UserId = 1u, ProductId = 102u, Rating = 3f },
			new ProductRating { UserId = 2u, ProductId = 101u, Rating = 4f },
			new ProductRating { UserId = 2u, ProductId = 103u, Rating = 2f },
			new ProductRating { UserId = 3u, ProductId = 104u, Rating = 5f }
		};

		var dataView = context.Data.LoadFromEnumerable(trainingData);

		// Configuração corrigida
		var options = new MatrixFactorizationTrainer.Options
		{
			MatrixColumnIndexColumnName = nameof(ProductRating.UserId),
			MatrixRowIndexColumnName = nameof(ProductRating.ProductId),
			LabelColumnName = nameof(ProductRating.Rating),
			NumberOfIterations = 20,
			LearningRate = 0.01f,
			ApproximationRank = 16
		};

		var pipeline = context.Recommendation().Trainers.MatrixFactorization(options);

		try
		{
			var model = pipeline.Fit(dataView);

			// Criação do motor corrigida com input/output types
			var engine = context.Model.CreatePredictionEngine<ProductRating, PredictionResult>(model);

			var testSample = new ProductRating { UserId = 1u, ProductId = 103u };
			var prediction = engine.Predict(testSample);

			Console.WriteLine($"Score previsto: {prediction.Score}");
		}
		catch (Exception ex)
		{
			Console.WriteLine($"Erro: {ex.Message}");
		}
	}
}