import React, { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown"; 

function App() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);


  useEffect(() => {
    document.body.style.backgroundColor = "#121212";
    document.body.style.color = "#ffffff";
    document.body.style.margin = 0;
    document.body.style.fontFamily = "Arial, sans-serif";
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResponse("");

    try {
      const res = await fetch("http://127.0.0.1:8000/api/query/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Server error: ${res.status} - ${errText}`);
      }

      const data = await res.json();
      setResponse(data.answer || "No response received from backend.");
    } catch (error) {
      console.error("Error:", error);
      setResponse("Error: " + error.message);
    }

    setLoading(false);
  };

  return (
    <div
      style={{
        maxWidth: "600px",
        margin: "50px auto",
        textAlign: "center",
        padding: "20px",
        borderRadius: "10px",
        backgroundColor: "#1e1e1e",
        boxShadow: "0 0 15px rgba(0,0,0,0.5)",
      }}
    >
      <h1 style={{ color: "#28a745" }}>RPulse</h1>

      <form onSubmit={handleSubmit}>
        <textarea
          rows="4"
          cols="50"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask your question..."
          required
          style={{
            width: "100%",
            padding: "10px",
            borderRadius: "5px",
            backgroundColor: "#2a2a2a",
            color: "#ffffff",
            border: "1px solid #333",
            resize: "none",
          }}
        />
        <br />
        <button
          type="submit"
          disabled={loading}
          style={{
            marginTop: "10px",
            padding: "10px 20px",
            borderRadius: "5px",
            backgroundColor: "#28a745",
            color: "#ffffff",
            border: "none",
            cursor: "pointer",
          }}
        >
          {loading ? "Processing..." : "Submit"}
        </button>
      </form>

      {response && (
        <div
          style={{
            marginTop: "20px",
            textAlign: "left",
            backgroundColor: "#2a2a2a",
            color: "#ffffff",
            padding: "15px",
            borderRadius: "5px",
          }}
        >
          <h3 style={{ color: "#28a745" }}>Response:</h3>
          <ReactMarkdown
            children={response}
            components={{
              p: ({ node, ...props }) => <p style={{ margin: "5px 0" }} {...props} />,
              li: ({ node, ...props }) => (
                <li style={{ marginBottom: "5px" }} {...props} />
              ),
              h1: ({ node, ...props }) => <h1 style={{ color: "#28a745" }} {...props} />,
              h2: ({ node, ...props }) => <h2 style={{ color: "#28a745" }} {...props} />,
              strong: ({ node, ...props }) => <strong style={{ color: "#28a745" }} {...props} />,
            }}
          />
        </div>
      )}
    </div>
  );
}

export default App;