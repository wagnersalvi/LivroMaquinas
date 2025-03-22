using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

public class HealthData
{
	public float Symptom1 { get; set; }
	public float Symptom2 { get; set; }
	public bool HasDisease { get; set; }
}

public class HealthPrediction
{
	[ColumnName("PredictedLabel")]
	public bool PredictedLabel { get; set; }
}

class Program
{
	static void Main(string[] args)
	{
		var mlContext = new MLContext();

		// Dados fictícios para diagnóstico de doenças
		var trainData = new List<HealthData>
		{
			new HealthData { Symptom1 = 0.8f, Symptom2 = 0.6f, HasDisease = true },
			new HealthData { Symptom1 = 0.2f, Symptom2 = 0.4f, HasDisease = false },
			new HealthData { Symptom1 = 0.7f, Symptom2 = 0.9f, HasDisease = true },
			new HealthData { Symptom1 = 0.3f, Symptom2 = 0.2f, HasDisease = false }
		};

		var trainingData = mlContext.Data.LoadFromEnumerable(trainData);

		// Pipeline de Random Forest
		var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "Symptom1", "Symptom2" })
			.Append(mlContext.BinaryClassification.Trainers.FastForest(labelColumnName: "HasDisease", featureColumnName: "Features"));

		// Treinando o modelo
		var model = pipeline.Fit(trainingData);

		var predictionEngine = mlContext.Model.CreatePredictionEngine<HealthData, HealthPrediction>(model);

		var newHealthData = new HealthData { Symptom1 = 0.9f, Symptom2 = 0.7f };
		var prediction = predictionEngine.Predict(newHealthData);

		Console.WriteLine($"Sintomas: {newHealthData.Symptom1}, {newHealthData.Symptom2}");
		Console.WriteLine($"Doença detectada? {(prediction.PredictedLabel ? "Sim" : "Não")}");
	}
}