async function ask() {
  const question = document.getElementById("question").value;
  const responseDiv = document.getElementById("response");
  responseDiv.innerText = "⏳ 正在生成回答...";
  const res = await fetch("http://127.0.0.1:8002/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question })
  });
  const data = await res.json();
  responseDiv.innerText = "🧠 Chatbot 回答：\n" + data.answer;
}
