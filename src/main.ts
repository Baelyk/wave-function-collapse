import SimpleKnotUrl from "./assets/SimpleKnot.png";
import { wave_function_collapse } from "../algorithm/pkg/";
(async () => {
	console.log("Hello from js!");
	console.log(SimpleKnotUrl);
	const imgBytes = new Uint8Array(
		await (await fetch(SimpleKnotUrl)).arrayBuffer(),
	);
	console.log(imgBytes);
	const result = wave_function_collapse(
		imgBytes,
		3,
		3,
		100,
		100,
		false,
		false,
		18,
	);
	const blob = new Blob([result], { type: "image/png" });
	const imgUrl = window.URL.createObjectURL(blob);
	const img = new Image();
	img.src = imgUrl;
	document.body.appendChild(img);
	console.log(result);
})();
