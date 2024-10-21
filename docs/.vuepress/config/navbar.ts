import { navbar } from "vuepress-theme-hope";

export default navbar([
    "/",
    {
        text: "Blog",
        icon: "blog",
        link: "/blog/",
        // TODO: drop-down menu
    },
    {
        text: "About",
        icon: "address-card",
        link: "about.html"
    }
]);
