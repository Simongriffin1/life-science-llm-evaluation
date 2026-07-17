import { expect, test } from "@playwright/test";

test.describe("BioLit console smoke", () => {
  test("retrieve view renders", async ({ page }) => {
    await page.goto("/retrieve");
    await expect(page.getByRole("heading", { name: "Retrieve" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Search" })).toBeVisible();
    await expect(page.getByText("BioLit")).toBeVisible();
  });

  test("evaluate view renders", async ({ page }) => {
    await page.goto("/evaluate");
    await expect(page.getByRole("heading", { name: "Evaluate" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Leaderboard" })).toBeVisible();
  });

  test("hypothesize view renders", async ({ page }) => {
    await page.goto("/hypothesize");
    await expect(page.getByRole("heading", { name: "Hypothesize" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Launch" })).toBeVisible();
    await expect(page.getByText("Ranked by Elo")).toBeVisible();
  });
});
