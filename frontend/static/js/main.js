document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultDisplay = document.getElementById('result-display');
    const chartContainer = document.getElementById('chart-container');
    let probabilityChart = null;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state
        resultDisplay.innerHTML = `
            <div style="text-align: center;">
                <div class="loader" style="border: 4px solid rgba(255,255,255,0.1); border-top: 4px solid var(--primary); border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 1rem;"></div>
                <p>Analyzing profile metrics...</p>
            </div>
        `;
        chartContainer.style.display = 'none';

        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.success) {
                const isPlaced = result.prediction === 1;
                const prob = (result.probability * 100).toFixed(1);
                
                // Update result display
                resultDisplay.innerHTML = `
                    <div style="text-align: center; animation: fadeIn 0.5s ease-out;">
                        <h3 style="color: var(--text-muted); margin-bottom: 0.5rem; font-size: 1.1rem;">Placement Probability</h3>
                        <div style="font-size: 4rem; font-weight: 700; background: linear-gradient(to bottom, #fff, ${isPlaced ? '#4ade80' : '#f87171'}); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">${prob}%</div>
                        <div class="status-badge ${isPlaced ? 'status-placed' : 'status-not-placed'}">
                            ${isPlaced ? '✓ Highly Probable' : '⚠ Limited Probability'}
                        </div>
                        <p style="margin-top: 1rem; color: var(--text-muted); line-height: 1.6; font-size: 0.95rem;">
                            Based on your profile analysis, you have a <strong>${prob}%</strong> chance of being placed in the current campus recruitment cycle.
                        </p>
                    </div>
                `;

                // Update Chart
                chartContainer.style.display = 'block';
                updateChart(result.probability);
            } else {
                resultDisplay.innerHTML = `<p style="color: #f87171;">Error: ${result.error}</p>`;
            }
        } catch (error) {
            console.error('Error:', error);
            resultDisplay.innerHTML = `<p style="color: #f87171;">Connection error. Is the backend running?</p>`;
        }
    });

    function updateChart(probability) {
        const ctx = document.getElementById('probabilityChart').getContext('2d');
        const data = {
            labels: ['Success Probability', 'Gap'],
            datasets: [{
                data: [probability, 1 - probability],
                backgroundColor: ['#6366f1', 'rgba(255, 255, 255, 0.05)'],
                borderWidth: 0,
                hoverOffset: 4
            }]
        };

        if (probabilityChart) {
            probabilityChart.destroy();
        }

        probabilityChart = new Chart(ctx, {
            type: 'doughnut',
            data: data,
            options: {
                cutout: '82%',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: false }
                },
                animation: {
                    animateRotate: true,
                    duration: 2000,
                    easing: 'easeOutQuart'
                }
            }
        });
    }
});

// Add spin animation to head via JS
const style = document.createElement('style');
style.textContent = `
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
`;
document.head.appendChild(style);
