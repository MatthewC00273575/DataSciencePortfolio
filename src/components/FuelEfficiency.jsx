import React from 'react';

const FuelEfficiency = () => (
    <section id="fuel-efficiency">
        <h1>Predicting Fuel Efficiency of Vehicles</h1>
        <h2>Overview</h2>
        <p>This project predicts a car’s fuel efficiency (MPG) using engine size, cylinders, drive type, and fuel type.</p>

        <h2>Methodology</h2>
        <h3>Linear Regression</h3>
        <img src="/images/LinearRegression.png" alt="Linear Regression" />
        <p>Initial models using engine displacement provided a basic fit but lacked high accuracy.</p>

        <h2>Decision Tree Regression</h2>
        <img src="/images/featureImportance.png" alt="Feature Importance" />
        <p>Engine displacement emerged as a primary predictor. A Decision Tree model captured more complex relationships.</p>

        <h3>Final Output</h3>
        <img src="/images/DecisionTreeOutput.png" alt="Decision Tree Output" />
        <p>The model used multiple predictors like engine displacement, cylinders, drive type, and fuel type for more accurate MPG predictions.</p>

        <h2>Technology Used</h2>
        <ul>
            <li>Pandas</li>
            <li>Scikit-Learn</li>
            <li>Matplotlib</li>
            <li>Numpy</li>
        </ul>
    </section>
);

export default FuelEfficiency;
