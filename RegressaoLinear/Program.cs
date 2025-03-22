using Microsoft.ML;
using RegressaoLinear;

MLContext contextoML = new MLContext();

// Importar ou criar dados de treinamento
DadosImovel[] dadosCasas = new DadosImovel[]
{
	new DadosImovel { Tamanho = 50.0F, Preco = 150.0F },
	new DadosImovel { Tamanho = 75.0F, Preco = 225.0F },
	new DadosImovel { Tamanho = 100.0F, Preco = 300.0F },
	new DadosImovel { Tamanho = 125.0F, Preco = 375.0F },
	new DadosImovel { Tamanho = 150.0F, Preco = 450.0F },
	new DadosImovel { Tamanho = 175.0F, Preco = 525.0F },
	new DadosImovel { Tamanho = 200.0F, Preco = 600.0F }
}; 
IDataView dadosTreinamento = contextoML.Data.LoadFromEnumerable(dadosCasas);

// Especificar pipeline de preparação de dados e treinamento do modelo
var pipeline = contextoML.Transforms.Concatenate("Features", new[] { "Tamanho" })
				.Append(contextoML.Regression.Trainers.LbfgsPoissonRegression(labelColumnName: "Preco")); 

// Treinar modelo
var modelo = pipeline.Fit(dadosTreinamento);


// Fazer uma previsão
var motorPrevisao = contextoML.Model.CreatePredictionEngine<DadosImovel, PredicaoImovel>(modelo);

for (int i = 50; i <= 250; i += 25)
{
	var dadoTeste = new DadosImovel { Tamanho = i };
	var previsao = motorPrevisao.Predict(dadoTeste);
	Console.WriteLine($"Tamanho: {i} m², Preço previsto: R${previsao.PrecoPrevisao:F2}");
}