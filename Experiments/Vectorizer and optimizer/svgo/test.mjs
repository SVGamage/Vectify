import { optimize } from "svgo";
import fs from "fs";

const svgContent = fs.readFileSync("../inputs/output.svg", "utf-8");

// Optimize the SVG
const result = optimize(svgContent, {
  // Customize optimization options here
  plugins: [
    { name: "convertPathData", params: { precision: 6 } }, // Adjust precision as needed
    { name: "mergePaths" },
  ],
});

// Save the optimized SVG
fs.writeFileSync("./output/output_enhanced.svg", result.data);

console.log("SVG optimized successfully!");
