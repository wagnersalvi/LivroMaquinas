using Clustering;
using Microsoft.ML;

class Program
{
	static void Main()
	{
		var context = new MLContext();
		var dados = new[]
		{
			new Cliente { GastoMensal = 200, FrequenciaCompras = 5 },
			new Cliente { GastoMensal = 50, FrequenciaCompras = 2 },
			new Cliente { GastoMensal = 1000, FrequenciaCompras = 10 },
			new Cliente { GastoMensal = 150, FrequenciaCompras = 3 }
		};

		var dataView = context.Data.LoadFromEnumerable(dados);
		var pipeline = context.Transforms.Concatenate("Features", "GastoMensal", "FrequenciaCompras")
						.Append(context.Clustering.Trainers.KMeans("Features", numberOfClusters: 3));

		var modelo = pipeline.Fit(dataView);
		var predictor = context.Model.CreatePredictionEngine<Cliente, ClusterPrediction>(modelo);

		var novoCliente = new Cliente { GastoMensal = 120, FrequenciaCompras = 4 };
		var resultado = predictor.Predict(novoCliente);

		Console.WriteLine($"O cliente foi classificado no cluster: {resultado.Cluster}");
	}
}