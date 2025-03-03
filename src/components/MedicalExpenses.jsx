import React from 'react';

const MedicalExpenses = () => (
    <section id="medical-expenses">
        <h1>Predicting Medical Expenses</h1>
        <h2>Overview</h2>
        <p>This project predicts an individual’s medical expenses using factors like age, BMI, smoking status, and region.</p>

        <h2>Methodology</h2>
        <h3>Random Forest Algorithm</h3>
        <img src="/images/FeatureImportanceRF.png" alt="Feature Importance" />
        <p>BMI, smoking status, and age were identified as the most influential factors in predicting medical costs.</p>

        <h3>Final Output</h3>
        <img src="/images/RFoutput.png" alt="Random Forest Output" />
        <p>The model captured complex interactions between demographic and lifestyle factors, leading to more accurate predictions.</p>

        <h2>Technology Used</h2>
        <ul>
            <li>Pandas</li>
            <li>Scikit-Learn</li>
            <li>Matplotlib</li>
            <li>Numpy</li>
        </ul>
    </section>
);

export default MedicalExpenses;
