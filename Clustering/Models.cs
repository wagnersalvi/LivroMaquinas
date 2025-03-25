using Microsoft.ML.Data;

namespace Clustering;

class Cliente
{
	[LoadColumn(0)] public float GastoMensal;
	[LoadColumn(1)] public float FrequenciaCompras;
}

class ClusterPrediction
{
	[ColumnName("PredictedLabel")] public uint Cluster;
}
