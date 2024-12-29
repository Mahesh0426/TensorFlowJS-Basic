// Create and print the tensor
const data = tf.tensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
// data.print();
// console.log(data.toString());
// console.log(data);

const values = [];
for (let i = 0; i < 30; i++) {
  values[i] = Math.random() * 100;
}

const shape = [2, 5, 3];
const data1 = tf.tensor3d(values, shape, "int32");
data1.print();
console.log(data1.toString());

// Update the div with the result
document.getElementById("micro-out-div").innerText = data.toString();
