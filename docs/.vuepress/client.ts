import { defineClientConfig } from "vuepress/client"
import { defineMermaidConfig } from "vuepress-plugin-md-enhance/client";

export default defineClientConfig({
  enhance: ({ app }) => {
    defineMermaidConfig({
      themeVariables: {
        fontFamily: "code"
      },
    });
  },
});
