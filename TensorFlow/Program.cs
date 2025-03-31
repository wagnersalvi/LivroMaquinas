using Microsoft.ML;
using TensorFlow;

class Program
{
	static void Main(string[] args)
	{
		// 1. Preparando os dados de treinamento
		var vendasAnuais = new List<VendaAnual>
		{
			new VendaAnual { Ano = 2015, ValorVendas = 50000 },
			new VendaAnual { Ano = 2016, ValorVendas = 52000 },
			new VendaAnual { Ano = 2017, ValorVendas = 53000 },
			new VendaAnual { Ano = 2018, ValorVendas = 55000 },
			new VendaAnual { Ano = 2019, ValorVendas = 58000 },
			new VendaAnual { Ano = 2020, ValorVendas = 60000 },
			new VendaAnual { Ano = 2021, ValorVendas = 63000 }
		};

		// 2. Criando um contexto do ML.NET
		var mlContext = new MLContext();

		// 3. Convertendo os dados para o formato que o ML.NET pode usar
		var vendasAnuaisData = mlContext.Data.LoadFromEnumerable(vendasAnuais);

		// 4. Definindo o pipeline de transformação e treinamento
		var pipeline = mlContext.Transforms.Concatenate("Features", "Ano")
			.Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "ValorVendas", maximumNumberOfIterations: 100));

		// 5. Treinando o modelo
		var modelo = pipeline.Fit(vendasAnuaisData);

		// 6. Usando o modelo para prever as vendas de 2025
		var previsao = modelo.Transform(vendasAnuaisData);

		// 7. Obtendo as previsões e mostrando o resultado
		var vendasPrevistas = mlContext.Data.CreateEnumerable<VendaAnual>(previsao, reuseRowObject: false).ToList();
		var previsao2025 = vendasPrevistas.Last().ValorVendas;

		Console.WriteLine($"Previsão de vendas para 2025: {previsao2025} reais");
	}
}
