import React from "react";
import {
  BrowserRouter as Router,
  Route,
  Routes,
  NavLink,
} from "react-router-dom";
import Home from "./components/Home";
import Resume from "./components/Resume";
import Portfolio from "./components/Portfolio";
import Skills from "./components/Skills";
import FuelEfficiency from "./components/FuelEfficiency";
import MedicalExpenses from "./components/MedicalExpenses";
import BayesianClassification from "./components/BayesianClassification";
import KMeansClustering from "./components/KMeansClustering";
import SupportVectorMachines from "./components/SupportVectorMachines";
import KNearestNeighbors from "./components/KNearestNeighbors";
import ZeroCodeRNN from "./components/ZeorCodeRNN";
import CNN from "./components/CNN";
import RNN from "./components/RNN";
import "./App.css";

const App = () => {
  return (
    <Router>
      <header>
        <nav>
          <ul>
            <li>
              <NavLink to="/" end>
                Home
              </NavLink>
            </li>
            <li>
              <NavLink to="/resume">Resume</NavLink>
            </li>
            <li>
              <NavLink to="/portfolio">Portfolio</NavLink>
            </li>
            <li>
              <NavLink to="/skills">Skills</NavLink>
            </li>
          </ul>
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/resume" element={<Resume />} />
          <Route path="/portfolio" element={<Portfolio />} />
          <Route path="/skills" element={<Skills />} />
          <Route path="/fuel-efficiency" element={<FuelEfficiency />} />
          <Route path="/medical-expenses" element={<MedicalExpenses />} />
          <Route
            path="/bayesian-classification"
            element={<BayesianClassification />}
          />
          <Route path="/k-means-clustering" element={<KMeansClustering />} />
          <Route
            path="/support-vector-machines"
            element={<SupportVectorMachines />}
          />
          <Route path="/k-nearest-neighbors" element={<KNearestNeighbors />} />
          <Route path="/zero-code-rnn" element={<ZeroCodeRNN />} />
          <Route path="/rnn" element={<RNN />} />
          <Route path="/cnn" element={<CNN />} />
        </Routes>
      </main>
      <footer>
        <p>&copy; 2024 Matthew Ufumeli. All rights reserved.</p>
      </footer>
    </Router>
  );
};

export default App;
