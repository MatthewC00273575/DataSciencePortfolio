import React from 'react';

const Skills = () => (
    <section id="skills">
        <div className="container">
            <h2>Skills</h2>

            {/* Programming Languages */}
            <h3>Programming Languages</h3>
            <div className="skill-block">
                <a href="https://www.python.org/" target="_blank" rel="noopener noreferrer">
                    <img src="/images/python.png" alt="Python" />
                    <p>Python</p>
                </a>
            </div>
            <div className="skill-block">
                <a href="https://www.java.com/" target="_blank" rel="noopener noreferrer">
                    <img src="/images/java.png" alt="Java" />
                    <p>Java</p>
                </a>
            </div>
            <div className="skill-block">
                <a href="https://isocpp.org/" target="_blank" rel="noopener noreferrer">
                    <img src="/images/c++.png" alt="C++" />
                    <p>C++</p>
                </a>
            </div>
            <div className="skill-block">
                <a href="https://dotnet.microsoft.com/en-us/languages/csharp" target="_blank" rel="noopener noreferrer">
                    <img src="/images/cSharp.png" alt="C#" />
                    <p>C#</p>
                </a>
            </div>
            <div className="skill-block">
                <a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript" target="_blank" rel="noopener noreferrer">
                    <img src="/images/javascript.png" alt="JavaScript" />
                    <p>JavaScript</p>
                </a>
            </div>
            <div className="skill-block">
                <a href="https://go.dev/" target="_blank" rel="noopener noreferrer">
                    <img src="/images/go.png" alt="Go" />
                    <p>Go</p>
                </a>
            </div>

            {/* Frameworks & Libraries */}
            <h3>Frameworks & Libraries</h3>
            <div className="skill-block">
                <a href="https://react.dev/" target="_blank" rel="noopener noreferrer">
                    <img src="/images/logo_dark.svg" alt="React" />
                    <p>React</p>
                </a>
            </div>
            <div className="skill-block">
                <a href="https://flutter.dev/" target="_blank" rel="noopener noreferrer">
                    <img src="/images/flutter.png" alt="Flutter" />
                    <p>Flutter</p>
                </a>
            </div>
            <div className="skill-block">
                <a href="https://fastapi.tiangolo.com/" target="_blank" rel="noopener noreferrer">
                    <img src="/images/fastapi.png" alt="FastAPI" />
                    <p>FastAPI</p>
                </a>
            </div>
            <div className="skill-block">
                <a href="https://flask.palletsprojects.com/" target="_blank" rel="noopener noreferrer">
                    <img src="/images/flask.png" alt="Flask" />
                    <p>Flask</p>
                </a>
            </div>

            {/* Tools & Technologies */}
            <h3>Tools & Technologies</h3>
            <div className="skill-block">
                <a href="https://git-scm.com/" target="_blank" rel="noopener noreferrer">
                    <img src="/images/git.png" alt="Git" />
                    <p>Git</p>
                </a>
            </div>
            <div className="skill-block">
                <a href="https://mariadb.org/" target="_blank" rel="noopener noreferrer">
                    <img src="/images/mariadb.png" alt="MariaDB" />
                    <p>MariaDB</p>
                </a>
            </div>
            <div className="skill-block">
                <a href="https://www.mysql.com/" target="_blank" rel="noopener noreferrer">
                    <img src="/images/mysql.png" alt="MySQL" />
                    <p>MySQL</p>
                </a>
            </div>
            <div className="skill-block">
                <a href="https://firebase.google.com/" target="_blank" rel="noopener noreferrer">
                    <img src="/images/firebase.jpeg" alt="Firebase" />
                    <p>Firebase</p>
                </a>
            </div>
        </div>
    </section>
);

export default Skills;