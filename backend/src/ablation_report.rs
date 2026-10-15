//! Shared native evidence tables for every paired ablation report.
use crate::Result;
use std::{fs, path::Path};
fn escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
pub fn decorate(path: &Path) -> Result<()> {
    let parent = path.parent().unwrap_or(Path::new("."));
    let mut html = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut content=String::from("<!-- RST19-NATIVE-EVIDENCE-START --><section class=\"native-evidence\"><h2>可复核实验记录</h2><p>候选提案、质量通过、注入回收和轨迹关联分项呈现。注入坐标是合成真值；这些记录不确认物理恒星身份，也不把无真值背景数量解释成 precision。</p>");
    for (folder, title) in [
        ("point", "点源提案与质量门"),
        ("motion", "冻结检测列表轨迹关联"),
        ("motion-proposal", "冻结像素运动提案"),
        ("crowded", "拥挤双源与局部去混叠"),
        ("motion-crossing", "交叉轨迹身份关联"),
    ] {
        let directory = parent.join("evidence").join(folder);
        if !directory.is_dir() {
            continue;
        }
        content.push_str(&format!("<article><h3>{title}</h3><p><a href=\"evidence/{folder}/manifest.json\">输入哈希、有效参数和限制</a></p>"));
        let mut files = fs::read_dir(&directory)
            .map_err(|e| e.to_string())?
            .map(|e| e.map(|e| e.path()).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>>>()?;
        files.sort();
        for file in files {
            let name = file
                .file_name()
                .and_then(|v| v.to_str())
                .ok_or("invalid evidence filename")?;
            if name.ends_with(".csv") {
                content.push_str(&format!("<details{}><summary>{}</summary><p><a href=\"evidence/{folder}/{}\">完整 CSV</a></p><div class=\"native-table\"><table>",if name=="summary.csv"{" open"}else{""},escape(name),escape(name)));
                let bytes = fs::read(&file).map_err(|e| e.to_string())?;
                if !bytes.is_empty() {
                    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
                    let mut reader = csv::Reader::from_reader(bytes);
                    let headers = reader
                        .headers()
                        .map_err(|e| format!("{}: {e}", file.display()))?
                        .clone();
                    content.push_str("<thead><tr>");
                    for field in &headers {
                        content.push_str(&format!("<th>{}</th>", escape(field)));
                    }
                    content.push_str("</tr></thead><tbody>");
                    let mut count = 0;
                    for record in reader.records() {
                        let record = record.map_err(|e| e.to_string())?;
                        count += 1;
                        if count <= 200 {
                            content.push_str("<tr>");
                            for field in &record {
                                content.push_str(&format!("<td>{}</td>", escape(field)));
                            }
                            content.push_str("</tr>");
                        }
                    }
                    content.push_str("</tbody>");
                    if count > 200 {
                        content.push_str(&format!(
                            "<caption>显示前 200 行；共 {count} 行，全部记录见 CSV。</caption>"
                        ));
                    }
                }
                content.push_str("</table></div></details>");
            } else if name.ends_with(".png") {
                content.push_str(&format!("<figure><img loading=\"lazy\" src=\"evidence/{folder}/{}\" alt=\"{}\"><figcaption>{}</figcaption></figure>",escape(name),escape(name),escape(name)));
            }
        }
        content.push_str("</article>");
    }
    content.push_str("</section><!-- RST19-NATIVE-EVIDENCE-END -->");
    let start = "<!-- RST19-NATIVE-EVIDENCE-START -->";
    let end = "<!-- RST19-NATIVE-EVIDENCE-END -->";
    if let Some(a) = html.find(start) {
        if let Some(b) = html[a..].find(end) {
            html.replace_range(a..a + b + end.len(), &content);
        } else {
            html.push_str(&content);
        }
    } else {
        let at = html.find("</body>").unwrap_or(html.len());
        html.insert_str(at, &content);
    }
    if !html.contains("id=\"rst19-native-evidence-style\"") {
        let style="<style id=\"rst19-native-evidence-style\">.native-evidence{max-width:1180px;margin:3rem auto;padding:1.5rem;color:#162235;font:16px/1.65 system-ui,sans-serif;background:#f7f4ed}.native-evidence article{margin:2rem 0;border-top:2px solid #263950;padding-top:1rem}.native-evidence a{color:#136b77}.native-evidence details{margin:1rem 0}.native-evidence summary{cursor:pointer;font-weight:600}.native-table{overflow:auto;max-height:560px;border:1px solid #cbd1d0}.native-evidence table{border-collapse:collapse;font-size:12px;white-space:nowrap}.native-evidence th{position:sticky;top:0;background:#263950;color:white;text-align:left}.native-evidence th,.native-evidence td{padding:8px 10px;border-bottom:1px solid #d6dbd7}.native-evidence tr:nth-child(even){background:#edf0e9}.native-evidence img{max-width:100%;height:auto}.native-evidence figcaption{color:#576778;font-size:13px}</style>";
        let at = html.find("</head>").unwrap_or(0);
        html.insert_str(at, style);
    }
    fs::write(path, html).map_err(|e| e.to_string())
}
