import React from 'react';

const Contact = () => (
    <section id="contact">
        <div className="container">
            <h2>Contact Me</h2>
            <form action="mailto:ufumeli2@gmail.com" method="post" encType="text/plain">
                <label htmlFor="name">Name:</label>
                <input type="text" id="name" name="name" required />
                <label htmlFor="email">Email:</label>
                <input type="email" id="email" name="email" required />
                <label htmlFor="message">Message:</label>
                <textarea id="message" name="message" rows="4" required></textarea>
                <button type="submit">Send Message</button>
            </form>
        </div>
    </section>
);

export default Contact;
