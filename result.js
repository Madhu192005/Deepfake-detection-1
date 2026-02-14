const raw = sessionStorage.getItem("deepfakeResult");
if (!raw) { window.location.href = "upload.html"; }
const data = JSON.parse(raw);
const isDocument = data.detection_mode === "document";

document.getElementById("subTitle").textContent = isDocument
  ? "Document authenticity analysis complete" : "AI-powered deepfake analysis complete";

const badge = document.getElementById("verdictBadge");
const verdictIcon = document.getElementById("verdictIcon");
const verdictText = document.getElementById("verdictText");

if (isDocument) {
  const v = data.overall_verdict || "";
  if (v.includes("FAKE") || v.includes("SUSPICIOUS")) {
    badge.classList.add("fake"); verdictIcon.textContent = "⚠️"; verdictText.textContent = "SUSPICIOUS DOCUMENT";
  } else {
    badge.classList.add("real"); verdictIcon.textContent = "✅"; verdictText.textContent = "LIKELY AUTHENTIC";
  }
} else {
  if (data.label === "FAKE") {
    badge.classList.add("fake"); verdictIcon.textContent = "⚠️"; verdictText.textContent = "DEEPFAKE DETECTED";
  } else {
    badge.classList.add("real"); verdictIcon.textContent = "✅"; verdictText.textContent = "AUTHENTIC MEDIA";
  }
}

if (isDocument) {
  document.getElementById("stat1Value").textContent = (data.ai_analysis?.ai_score || 0).toFixed(1) + "%";
  document.getElementById("stat1Label").textContent = "AI Text Score";
  document.getElementById("stat2Value").textContent = (data.tamper_analysis?.tamper_score || 0).toFixed(1) + "%";
  document.getElementById("stat2Label").textContent = "Tamper Score";
  document.getElementById("stat3Value").textContent = (data.overall_fake_score || 0).toFixed(1) + "%";
  document.getElementById("stat3Label").textContent = "Overall Score";
} else {
  document.getElementById("stat1Value").textContent = (data.fake_probability || 0).toFixed(1) + "%";
  document.getElementById("stat2Value").textContent = (data.real_probability || 0).toFixed(1) + "%";
  document.getElementById("stat3Value").textContent = (data.confidence || 0).toFixed(1) + "%";
}

const fileInfoEl = document.getElementById("fileInfo");
if (data.filename) {
  let icon = data.file_type === "PDF" ? "📕" : data.file_type === "DOCX" ? "📘" : data.type === "video" ? "🎬" : "🖼️";
  fileInfoEl.textContent = `${icon} ${data.filename}`;
  if (data.frames_analyzed) fileInfoEl.textContent += ` • ${data.frames_analyzed} frames`;
  if (data.page_count) fileInfoEl.textContent += ` • ${data.page_count} pages`;
}

if (isDocument) {
  document.getElementById("docSection").style.display = "block";
  const ai = data.ai_analysis || {};
  const tamper = data.tamper_analysis || {};
  const meta = tamper.metadata || {};
  document.getElementById("aiScore").textContent = (ai.ai_score || 0) + "%";
  document.getElementById("wordCount").textContent = ai.word_count || "0";
  document.getElementById("aiVerdict").textContent = ai.verdict || "Unknown";
  document.getElementById("patternsFound").textContent = ai.ai_patterns_found?.length > 0 ? ai.ai_patterns_found.join(", ") : "None";
  document.getElementById("tamperScore").textContent = (tamper.tamper_score || 0) + "%";
  document.getElementById("docAuthor").textContent = meta.author || "Unknown";
  document.getElementById("docCreated").textContent = meta.created || "Unknown";
  const issuesList = document.getElementById("issuesList");
  const issues = tamper.issues || [];
  issuesList.innerHTML = issues.length === 0 ? '<li style="color:#38a169;">✅ No issues found</li>' : issues.map(i => `<li>⚠️ ${i}</li>`).join("");
}

const ctx = document.getElementById("aiGraph").getContext("2d");
if (isDocument) {
  new Chart(ctx, { type:"bar", data:{ labels:["AI Text Score","Tamper Score","Overall Score"],
    datasets:[{ data:[data.ai_analysis?.ai_score||0, data.tamper_analysis?.tamper_score||0, data.overall_fake_score||0],
    backgroundColor:["rgba(255,122,0,0.7)","rgba(229,62,62,0.7)","rgba(113,53,0,0.7)"],
    borderColor:["#ff7a00","#e53e3e","#713500"], borderWidth:2, borderRadius:8 }]},
    options:{ responsive:true, plugins:{legend:{display:false}}, scales:{y:{min:0,max:100,ticks:{callback:v=>v+"%"}}}}});
} else if (data.type === "video" && data.frame_labels?.length > 0) {
  new Chart(ctx, { type:"line", data:{ labels:data.frame_labels,
    datasets:[{ label:"Fake % per Frame", data:data.frame_values, borderColor:"#ff7a00",
    backgroundColor:"rgba(255,122,0,0.15)", fill:true, tension:0.4, pointRadius:5 }]},
    options:{ responsive:true, scales:{y:{min:0,max:100,ticks:{callback:v=>v+"%"}}}}});
  if (data.frames_analyzed) document.getElementById("framesInfo").textContent = `Analyzed ${data.frames_analyzed} frames`;
} else {
  new Chart(ctx, { type:"bar", data:{ labels:["Real Probability","Fake Probability"],
    datasets:[{ data:[data.real_probability||0, data.fake_probability||0],
    backgroundColor:["rgba(56,161,105,0.7)","rgba(229,62,62,0.7)"],
    borderColor:["#38a169","#e53e3e"], borderWidth:2, borderRadius:8 }]},
    options:{ responsive:true, plugins:{legend:{display:false}}, scales:{y:{min:0,max:100,ticks:{callback:v=>v+"%"}}}}});
}