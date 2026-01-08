def nutrition_recommendation_agent(food_records):
    """
    输入：多张图片识别后的 food + nutrition
    输出：营养汇总 + 推荐建议
    """

    total = {
        "calories_kcal": 0.0,
        "protein_g": 0.0,
        "fat_g": 0.0,
        "carbs_g": 0.0
    }

    valid_items = 0

    for item in food_records:
        nutrition = item.get("nutrition")
        if not nutrition:
            continue

        valid_items += 1
        for k in total:
            total[k] += nutrition.get(k, 0.0)

    # -------- 决策规则 --------
    recommendations = []
    warnings = []

    if valid_items == 0:
        return {
            "total_nutrition": total,
            "recommendations": ["No nutrition data available to make suggestions."],
            "warnings": []
        }

    # 热量
    if total["calories_kcal"] > 900:
        recommendations.append("Consider light or low-calorie foods (e.g., salad, soup)")
    else:
        recommendations.append("Calorie intake is acceptable")

    # 蛋白
    if total["protein_g"] < 40:
        recommendations.append("Add high-protein foods (e.g., eggs, chicken, tofu)")
    else:
        recommendations.append("Protein intake is sufficient")

    # 脂肪
    if total["fat_g"] > 35:
        warnings.append("Avoid fried or high-fat foods")

    return {
        "total_nutrition": total,
        "recommendations": recommendations,
        "warnings": warnings
    }
