document.addEventListener('DOMContentLoaded', async () => {
    const modelSelect = document.getElementById('modelSelect');

    const fetchModelData = async (model) => {
        const response = await fetch(`/train?model=${model}`, { method: 'POST' });
        const data = await response.json();

        // Update Metrics
        document.getElementById('accuracy').innerText = data.metrics.accuracy;
        document.getElementById('dataPoints').innerText = data.metrics.data_points;
        document.getElementById('trainingTime').innerText = data.metrics.training_time;

        // Update Plots
        if (data.plots.residual_plot) {
            document.getElementById('residualPlot').src = `data:image/png;base64,${data.plots.residual_plot}`;
        }
        if (data.plots.error_histogram) {
            document.getElementById('errorHistogram').src = `data:image/png;base64,${data.plots.error_histogram}`;
        }
        if (data.plots.prediction_vs_actual) {
            document.getElementById('predictionVsActual').src = `data:image/png;base64,${data.plots.prediction_vs_actual}`;
        }
        if (data.plots.feature_importance) {
            document.getElementById('featureImportance').src = `data:image/png;base64,${data.plots.feature_importance}`;
        }
    };

    // Fetch data for the default model
    fetchModelData(modelSelect.value);

    // Update data when a new model is selected
    modelSelect.addEventListener('change', (e) => {
        fetchModelData(e.target.value);
    });
});
