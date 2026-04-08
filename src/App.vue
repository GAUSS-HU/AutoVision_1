<script setup>
import { ref } from "vue"

const result = ref(null)
const loading = ref(false)
const error = ref("")

async function upload(files) {
  if (!files || files.length === 0) return

  loading.value = true
  error.value = ""
  result.value = null

  try {
    const fd = new FormData()
    for (const f of files) {
      fd.append("files", f)
    }

    const res = await fetch("http://127.0.0.1:8000/predict_batch", {
      method: "POST",
      body: fd
    })

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`)
    }

    result.value = await res.json()
  } catch (err) {
    error.value = "请求失败：" + err.message
    console.error(err)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div style="max-width: 1000px; margin: 30px auto; font-family: Arial, sans-serif;">
    <h2>AutoVision Food Agent</h2>
    <p>上传多张食物图片，查看识别结果、营养信息和推荐建议。</p>

    <input
      type="file"
      multiple
      accept="image/*"
      @change="e => upload(e.target.files)"
    />

    <p v-if="loading">Analyzing...</p>
    <p v-if="error" style="color: red;">{{ error }}</p>

    <div v-if="result" style="margin-top: 24px;">
      <div style="border: 1px solid #ddd; border-radius: 10px; padding: 12px; margin-bottom: 20px;">
        <h3>Recommendation Summary</h3>
        <div><b>Total calories:</b> {{ result.recommendation.total_nutrition.calories_kcal }}</div>
        <div><b>Total protein:</b> {{ result.recommendation.total_nutrition.protein_g }}</div>
        <div><b>Total fat:</b> {{ result.recommendation.total_nutrition.fat_g }}</div>
        <div><b>Total carbs:</b> {{ result.recommendation.total_nutrition.carbs_g }}</div>

        <h4>Recommendations</h4>
        <ul>
          <li v-for="(item, idx) in result.recommendation.recommendations" :key="idx">
            {{ item }}
          </li>
        </ul>

        <h4>Warnings</h4>
        <ul>
          <li v-for="(item, idx) in result.recommendation.warnings" :key="'w'+idx">
            {{ item }}
          </li>
        </ul>
      </div>

      <div
        v-for="(item, idx) in result.items"
        :key="idx"
        style="border: 1px solid #ddd; border-radius: 10px; padding: 12px; margin-bottom: 16px;"
      >
        <h3>Image {{ idx + 1 }}</h3>
        <div><b>Final class:</b> {{ item.final_class }}</div>
        <div><b>Agent reason:</b> {{ item.agent_reason }}</div>

        <div style="margin-top: 10px;">
          <h4>Nutrition</h4>
          <div v-if="item.nutrition">
            <div>Calories: {{ item.nutrition.calories_kcal }}</div>
            <div>Protein: {{ item.nutrition.protein_g }}</div>
            <div>Fat: {{ item.nutrition.fat_g }}</div>
            <div>Carbs: {{ item.nutrition.carbs_g }}</div>
          </div>
          <div v-else>No nutrition data available.</div>
        </div>

        <div style="margin-top: 10px;">
          <h4>Top-5 Predictions</h4>
          <table border="1" cellspacing="0" cellpadding="6" style="border-collapse: collapse; width: 100%;">
            <thead>
              <tr>
                <th>#</th>
                <th>Class</th>
                <th>Prob</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(pred, i) in item.topk" :key="i">
                <td>{{ i + 1 }}</td>
                <td>{{ pred[0] }}</td>
                <td>{{ pred[1].toFixed(4) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</template>