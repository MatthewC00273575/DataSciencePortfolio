import React from 'react';
import { NavLink } from 'react-router-dom';

const Portfolio = () => (
    <section id="portfolio">
        <div className="container">
            <h2>Projects</h2>
            <div className="projects-list">
                <div className="project-item">
                    <h3>Secret Santa App (Flutter, Dart)</h3>
                    <p>Mobile app with Firebase Firestore for gift exchanges.</p>
                    <a href="https://github.com/MatthewC00273575/secret-santa" target="_blank" rel="noopener noreferrer">GitHub Link</a>
                </div> <br></br>
                <h2>Data Science </h2>
                <br></br>
                <div className="project-item">
                    <h3>Bayesian Classification</h3>
                    <p>Naive Bayes classification applied to the UCI Mushroom dataset.</p>
                    <NavLink to="/bayesian-classification">View Project Details</NavLink>
                </div>
                <div className="project-item">
                    <h3>K-Means Clustering</h3>
                    <p>Clustering customer spending patterns using the Wholesale dataset.</p>
                    <NavLink to="/k-means-clustering">View Project Details</NavLink>
                </div>
                <div className="project-item">
                    <h3>Support Vector Machines</h3>
                    <p>SVM model classifying wine quality from numerical attributes.</p>
                    <NavLink to="/support-vector-machines">View Project Details</NavLink>
                </div>
                <div className="project-item">
                    <h3>K-Nearest Neighbour</h3>
                    <p>KNN model classifying abalone age from physical measurements using the UCI Abalone Dataset.</p>
                    <NavLink to="/k-nearest-neighbors">View Project Details</NavLink>
                </div>

                <div className="project-item">
                    <h3>Chat-GPT RNN</h3>
                    <p>RNN model created only using chat-gpt using a custom created dataset.</p>
                    <NavLink to="/zero-code-rnn">View Project Details</NavLink>
                </div>
            </div>
        </div>
    </section>
);

export default Portfolio;