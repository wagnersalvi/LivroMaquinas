using Microsoft.ML;
using Microsoft.ML.Transforms;

using PredicaoCasa;


MLContext contextoML = new MLContext();

// Importar ou criar dados de treinamento
DadosImovel[] dadosCasas = new DadosImovel[]
{
			new DadosImovel { Tamanho = 110.0F, Preco = 120.0F },
			new DadosImovel { Tamanho = 190.0F, Preco = 230.0F },
			new DadosImovel { Tamanho = 280.0F, Preco = 300.0F },
			new DadosImovel { Tamanho = 340.0F, Preco = 370.0F }
};
IDataView dadosTreinamento = contextoML.Data.LoadFromEnumerable(dadosCasas);

// Visualizar os dados

var preview = contextoML.Data.CreateEnumerable<DadosImovel>(dadosTreinamento, reuseRowObject: false);
foreach (var casa in preview)
{
	Console.WriteLine($"Tamanho: {casa.Tamanho} m², Preço: R${casa.Preco}k");
}


// Especificar pipeline de preparação de dados e treinamento do modelo, primeiro troca os vazios ela média
var pipeline = contextoML.Transforms.ReplaceMissingValues(
	outputColumnName: "TamanhoProcessado",
	inputColumnName: "Tamanho",
	replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
.Append(contextoML.Transforms.Concatenate("Features", new[] { "TamanhoProcessado" }))
.Append(contextoML.Regression.Trainers.Sdca(labelColumnName: "Preco", maximumNumberOfIterations: 100));

//var pipeline = contextoML.Transforms.Concatenate("Features", new[] { "Caracteristica1", "Caracteristica2" })
//	.Append(contextoML.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Rotulo"));

// Treinar modelo
var modelo = pipeline.Fit(dadosTreinamento);

// Fazer uma previsão
var motorPrevisao = contextoML.Model.CreatePredictionEngine<DadosImovel, PredicaoImovel>(modelo);

var tamanhoParaPrever = new DadosImovel { Tamanho = 200.0F };
var previsaoPreco = motorPrevisao.Predict(tamanhoParaPrever);

Console.WriteLine($"Preço previsto para tamanho: {tamanhoParaPrever.Tamanho} " +
	$"m² = R${previsaoPreco.PrecoPrevisao:F2}k");

// Avaliar modelo
DadosImovel[] dadosTesteCasas = new DadosImovel[]
{
			new DadosImovel { Tamanho = 110.0F, Preco = 98.0F },
			new DadosImovel { Tamanho = 190.0F, Preco = 210.0F },
			new DadosImovel { Tamanho = 280.0F, Preco = 290.0F },
			new DadosImovel { Tamanho = 340.0F, Preco = 360.0F }
};
var dadosTesteCasasView = contextoML.Data.LoadFromEnumerable(dadosTesteCasas);
var dadosPrecoTeste = modelo.Transform(dadosTesteCasasView);
var metricas = contextoML.Regression.Evaluate(dadosPrecoTeste, labelColumnName: "Preco");

Console.WriteLine($"R²: {metricas.RSquared:F2}");
Console.WriteLine($"Erro RMS: {metricas.RootMeanSquaredError:F2}");
