# 需求文档

## 介绍

升级项目spring框架依赖的版本到spring6

## 需求

## 1. 背景介绍

- Spring 5 官方支持已停止，安全需要；
- 项目及技术升级(如Spring AI)的需要。

## 2. 升级步骤

**重要：** 以下升级步骤是全量通用需求，若项目不涉及该部分，跳过即可。
**重要：** 已经充分做了升级技术调研、并且升级步骤已完整验证，不需要做任何额外的调研，直接生成技术文档即可。
**重要：** 不需要生成research.md调研文档，直接生成技术文档。

### 2.1 切换编译环境

- Spring6/SpringBoot3将Java基线定在了JDK17，升级前提是JDK编译和运行环境都要在JDK17之上。
- 本地环境JDK17安装在 `/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home`

```bash
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home
export PATH=$JAVA_HOME/bin:$PATH
```

### 2.2 切换cicd环境

修改 `.gitlab-ci.yml` 文件，增加变量，指定运行环境为 jdk17：

```yaml
variables:
  TOOLS_SETTING_JDK: "jdk17"
```

### 2.3 校验服务运行环境

由于升级后只支持 JDK17 运行，不再支持 JDK8，所以在服务启动时校验版本。

**需要修改的脚本文件：**

- 服务启动脚本（如 `start.sh`、`{服务名}.sh` 等）
- 数据库和配置维护脚本（如 `core.sh`、`init.sh` 等）

#### 2.3.1 服务启动脚本调整

**原配置示例：**

```bash
JAVA_MAJOR_VERSION=$("$JAVACMD" -version 2>&1 | sed -E -n 's/.* version "([0-9]*).*$/\1/p')
echo "Java version: $JAVA_MAJOR_VERSION"
if [[ $JAVA_MAJOR_VERSION -ge 9 ]]; then
  JVM_OPTS="${JVM_OPTS} -Xlog:gc*:file=$APP_LOG_DIR/gc-%t.log:time,level:filecount=3,filesize=10m"
else
  JVM_OPTS="${JVM_OPTS} -Xloggc:$APP_LOG_DIR/gc-%t.log -XX:GCLogFileSize=10m -XX:+UseGCLogFileRotation"
  JVM_OPTS="${JVM_OPTS} -XX:NumberOfGCLogFiles=3 -XX:+PrintGCDetails -XX:+PrintGCDateStamps"
fi
```

**修改后示例：**

```bash
JAVA_MAJOR_VERSION=$("$JAVACMD" -version 2>&1 | sed -E -n 's/.* version "([0-9]*).*$/\1/p')
echo "Java version: $JAVA_MAJOR_VERSION"
if [[ $JAVA_MAJOR_VERSION -ge 17 ]]; then
  JVM_OPTS="${JVM_OPTS} -Xlog:gc*:file=$APP_LOG_DIR/gc-%t.log:time,level:filecount=3,filesize=10m"
else
  echo "Current Java Version: $JAVA_MAJOR_VERSION,Must be upgraded to JDK 17 or above."
  exit 1
fi
```

#### 2.3.2 数据库和配置维护脚本调整

**原配置示例：**

```bash
nacos_util_run_cmd="java -DconfigMergeStrategy=$config_merge_strategy -DnacosConfigPath=$base_dir/$nacos_config_path -DnacosConfigBackName=$5/$nacos_config_back_name -cp $classpath com.nacosutil.NacosConfig $2"
init_key_cmd="java -cp $key_classpath com.cloudwise.douc.commons.utils.GenKeyInit"
```

在 java 命令前修改，增加 JDK17 版本校验。

**注意：** 脚本会优先检查 `JAVA_17_HOME` 环境变量，如果未设置则使用 `JAVA_HOME`。

**修改后示例：**

```bash
######### Java 命令检查
if [ -n "$JAVA_17_HOME" ]; then
    if [ -x "$JAVA_17_HOME/jre/sh/java" ]; then
        JAVACMD="$JAVA_17_HOME/jre/sh/java"
    else
        JAVACMD="$JAVA_17_HOME/bin/java"
    fi
    if [ ! -x "$JAVACMD" ]; then
        echo "ERROR: JAVA_17_HOME is set to an invalid directory: $JAVA_17_HOME"
    fi
elif [ -n "$JAVA_HOME" ]; then
  if [ -x "$JAVA_HOME/jre/sh/java" ]; then
    JAVACMD="$JAVA_HOME/jre/sh/java"
  else
    JAVACMD="$JAVA_HOME/bin/java"
  fi
  if [ ! -x "$JAVACMD" ]; then
    echo "ERROR: JAVA_HOME is set to an invalid directory: $JAVA_HOME"
  fi
else
  JAVACMD="java"
  which java >/dev/null 2>&1 || echo "ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH."
fi

JAVA_MAJOR_VERSION=$("$JAVACMD" -version 2>&1 | sed -E -n 's/.* version "([0-9]*).*$/\1/p')
if [[ ${JAVA_MAJOR_VERSION:-0} -lt 17 ]]; then
  echo "Current Java Version: ${JAVA_MAJOR_VERSION:-unknown}, Must be upgraded to JDK 17 or above."
  exit 1
fi

nacos_util_run_cmd="$JAVACMD -DconfigMergeStrategy=$config_merge_strategy -DnacosConfigPath=$base_dir/$nacos_config_path -DnacosConfigBackName=$5/$nacos_config_back_name -cp $classpath com.nacosutil.NacosConfig $2"
init_key_cmd="$JAVACMD -cp $key_classpath com.cloudwise.douc.commons.utils.GenKeyInit"
```

### 2.4 服务代码适配

#### 2.4.1 依赖升级

升级 `inf-bom` 版本至 `3.0.2`，已将 spring、tomcat、jakartaEE 等版本调整到支持 Spring 6 和 Spring Boot 3 的版本。

```xml

<parent>
    <groupId>com.cloudwise</groupId>
    <artifactId>inf-bom</artifactId>
    <version>3.0.2</version>
    <relativePath/> <!-- lookup parent from repository -->
</parent>
```

#### 2.4.2 Java EE 切换到 Jakarta EE

升级后 Java EE 均切换到 Jakarta EE，对应的 class 的包名从 `javax.*` 变成了 `jakarta.*`。

**常见包名变更：**

- `javax.mail.internet.*` 修改为 `jakarta.mail.internet.*`
- `javax.servlet.*` 修改为 `jakarta.servlet.*`
- `javax.activation.*` 修改为 `jakarta.activation.*`
- `javax.websocket.*` 修改为 `jakarta.websocket.*`
- `javax.validation.*` 修改为 `jakarta.validation.*`
- `javax.annotation.*` 修改为 `jakarta.annotation.*`
- `javax.xml.soap.*` 修改为 `jakarta.xml.soap.*`

以下是一些常见的依赖变更和常见问题：

##### 2.4.2.1 douc-sdk-core 迁移

新增或编辑 properties 属性：

```xml

<douc-sdk-core.version>7.1.0</douc-sdk-core.version>
```

##### 2.4.2.2 portal-dubbo-api 迁移

新增或编辑 properties 属性：

```xml

<portal-dubbo-api.version>7.1.0</portal-dubbo-api.version>
```

##### 2.4.2.3 servlet-api 迁移

**删除以下依赖：**

```xml

<dependency>
    <groupId>javax.servlet</groupId>
    <artifactId>javax.servlet-api</artifactId>
</dependency>
```

**新增依赖：**

```xml

<dependency>
    <groupId>jakarta.servlet</groupId>
    <artifactId>jakarta.servlet-api</artifactId>
</dependency>
```

##### 2.4.2.4 shiro 不支持 JAKARTA

调整 shiro 依赖：

```xml

<dependency>
    <groupId>org.apache.shiro</groupId>
    <artifactId>shiro-spring</artifactId>
    <classifier>jakarta</classifier>
</dependency>
<dependency>
<groupId>org.apache.shiro</groupId>
    <artifactId>shiro-core</artifactId>
    <classifier>jakarta</classifier>
</dependency>
<dependency>
    <groupId>org.apache.shiro</groupId>
    <artifactId>shiro-web</artifactId>
    <classifier>jakarta</classifier>
</dependency>
```

##### 2.4.2.5 druid 的监控功能不支持 jakarta

**原代码：**

```java
import com.alibaba.druid.support.http.StatViewServlet;
```

**修改为：**

```java
import com.alibaba.druid.support.jakarta.StatViewServlet;
```

##### 2.4.2.6 javax.mail-api 不支持 jakarta

**原依赖：**

```xml

<dependency>
    <groupId>javax.mail</groupId>
    <artifactId>javax.mail-api</artifactId>
</dependency>
```

**修改为：**

```xml

<dependency>
    <groupId>jakarta.mail</groupId>
    <artifactId>jakarta.mail-api</artifactId>
</dependency>
```

##### 2.4.2.7 cn.hutool.extra.servlet.ServletUtil 不支持 Jakarta

**原代码：**

```java
import cn.hutool.extra.servlet.ServletUtil;
```

**修改为：**

```java
import cn.hutool.extra.servlet.JakartaServletUtil;
```

##### 2.4.2.8 javax.xml.soap.SOAPElement 不存在

导入依赖：

```xml

<dependency>
    <groupId>javax.xml.soap</groupId>
    <artifactId>javax.xml.soap-api</artifactId>
</dependency>
```

##### 2.4.2.9 validation 参数校验报错

**问题现象：**

```
jakarta.validation.NoProviderFoundException: Unable to create a Configuration, because no Jakarta Validation provider could be found. Add a provider like Hibernate Validator (RI) to your classpath.
```

**解决方案：**

增加依赖：

```xml

<dependency>
    <groupId>org.hibernate.validator</groupId>
    <artifactId>hibernate-validator</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-validation</artifactId>
    <exclusions>
        <exclusion>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-logging</artifactId>
        </exclusion>
    </exclusions>
</dependency>
```

##### 2.4.2.10 swagger 变更

**问题现象：**

```
java.lang.TypeNotPresentException: Type javax.servlet.http.HttpServletRequest not present
```

**解决方案：**

**移除以下依赖：**

```xml

<dependency>
    <groupId>com.github.xiaoymin</groupId>
    <artifactId>knife4j-spring-boot-starter</artifactId>
</dependency>
```

**新增依赖：**

```xml

<dependency>
    <groupId>com.github.xiaoymin</groupId>
    <artifactId>knife4j-openapi3-jakarta-spring-boot-starter</artifactId>
</dependency>
```

**移除以下代码（移除 swagger 全部内容）：**

```java

@Configuration
@EnableSwagger2 // 标记项目启用 Swagger API 接口文档
public class SwaggerConfiguration {
    @Bean
    public Docket createRestAPI() {
        return new Docket(DocumentationType.OAS_30)
                .apiInfo(apiInfo())
                .select().apis(RequestHandlerSelectors.withClassAnnotation(RestController.class))
                .paths(PathSelectors.any())
                .build();
    }

    private ApiInfo apiInfo() {
        return new ApiInfoBuilder()
                .title(StringUtils.defaultIfBlank(SpringUtil.getApplicationName(), "swagger-title"))
                .description(StringUtils.EMPTY).termsOfServiceUrl(StringUtils.EMPTY)
                .contact(new Contact(StringUtils.EMPTY, StringUtils.EMPTY, StringUtils.EMPTY))
                .version("1.0.0")
                .build();
    }
    .......................
}
```

**修改为：**

```java

@Configuration
public class SwaggerConfiguration {
    @Bean
    public OpenAPI projectOpenAPI() {
        return new OpenAPI()
                .info(new Info()
                        .title(StringUtils.defaultIfBlank(SpringUtil.getApplicationName(), "OpenAPI"))
                        .description("")
                        .version("1.0.0")
                        .contact(new Contact().name("").url("").email(""))
                );
    }

    @Bean
    public GroupedOpenApi apiGroup() {
        return GroupedOpenApi.builder()
                .group("api")
                .packagesToScan("com.cloudwise")
                .build();
    }
}
```

新增配置

```properties
springdoc.swagger-ui.enabled=true
springdoc.swagger-ui.urls[0].name=api
springdoc.swagger-ui.urls[0].url=/v3/api-docs
springdoc.swagger-ui.urls[1].name=dubbo
springdoc.swagger-ui.urls[1].url=/swagger-dubbo/api-docs
```

#### 2.4.3 对外提供的 SDK

**重要：** 为了保证兼容性，对外提供的 SDK，继续使用 JDK8 进行编译（inf-bom 可依赖 2.x）。

多个 SDK 可以支持不同的编译版本，GitLab CI/CD 配置示例：

```yaml
.deploy_douc_template:
  extends: .deploy_nexus_template
  only:
    - master
    - develop

# 发布对外sdk(8编译)
deploy-sdk-8:
  extends: .deploy_douc_template
  variables:
    TOOLS_SETTING_JDK: "jdk8"
  parallel:
    matrix:
      - PROJECT_MODULE:
          - sdk/apim-register-sdk
          - sdk/message-dubbo-api
          - sdk/openapi-api
# 发布对外sdk(17编译)
deploy-sdk-17:
  extends: .deploy_douc_template
  parallel:
    matrix:
      - PROJECT_MODULE:
          - sdk/douc-dubbo-api
```

#### 2.4.4 升级重要变更

##### 2.4.4.1 Spring Cloud Alibaba 配置方式变更

**重要：** 该版本起，Spring Cloud Alibaba 迁移到 `spring.config.import` 动态加载模式。

**原配置示例（移除 `spring.cloud.nacos.config.*` 相关配置）：**

```properties
# 导入nacos数据源
spring.cloud.nacos.config.extension-configs[0].data-id=portal-service-default.properties
spring.cloud.nacos.config.extension-configs[0].group=portal
spring.cloud.nacos.config.extension-configs[0].refresh=true
spring.cloud.nacos.config.extension-configs[1].data-id=portal-service.properties
spring.cloud.nacos.config.extension-configs[1].group=portal
spring.cloud.nacos.config.extension-configs[1].refresh=true
spring.cloud.nacos.config.shared-configs[0].data-id=commons.properties
spring.cloud.nacos.config.shared-configs[0].group=commons
spring.cloud.nacos.config.shared-configs[0].refresh=true
```

**新配置示例：**

1. 使用 `spring.config.import` 动态加载配置，多个使用数组或者逗号连接，优先级高的写后面
2. `optional:nacos:` 和 `nacos:` 是固定写法，`optional:` 代表配置是可选项

```properties
# 导入nacos数据源
spring.config.import[0]=nacos:commons.properties?group=commons&refreshEnabled=true
spring.config.import[1]=optional:nacos:portal-service-default.properties?group=portal&refreshEnabled=true
spring.config.import[2]=optional:nacos:portal-service.properties?group=portal&refreshEnabled=true
```

##### 2.4.4.2 配置文件 key 值变更

**重要：** 部分配置属性已变更，可参考提示按需修改。

**常见属性变更：**

- `spring.redis` 已移至 `spring.data.redis`。
- `spring.data.cassandra` 已移至 `spring.cassandra`。
- 移除了`spring.jpa.hibernate.use-new-id-generator`。
- `server.max.http.header.size` 已移至 `server.max-http-request-header-size`。
- 移除了对 `spring.security.saml2.relyingparty.registration.{id}.identity-provider` 的支持。

**识别废弃属性：**

在 `pom.xml` 中添加 `spring-boot-properties-migrator` 依赖，该依赖会在启动时生成并打印一份报告，列出已废弃的属性名称，并在运行时临时迁移这些属性。

**参考文档：** https://springdoc.cn/spring-boot-3-migration/

**示例报告：**

Property source 'applicationConfig: [classpath:/bootstrap.properties]':

```
        Key: spring.cloud.gateway.httpclient.connect-timeout
                Replacement: spring.cloud.gateway.server.webflux.httpclient.connect-timeout
        Key: spring.redis.database
                Replacement: spring.data.redis.database
        Key: spring.redis.timeout
                Replacement: spring.data.redis.timeout
        key: spring.cloud.gateway.httpclient.websocket.max-frame-payload-length
                Replacement: spring.cloud.gateway.server.webflux.httpclient.websocket.max-frame-payload-length
        key: spring.cloud.gateway.httpclient.connect-timeout
                Replacement: spring.cloud.gateway.server.webflux.httpclient.connect-timeout
```

##### 2.4.4.3 EnableAutoConfiguration 被移除

**重要：** 该版本起，彻底移除对 `org.springframework.boot.autoconfigure.EnableAutoConfiguration` 配置的支持，Spring
已经删除相关逻辑。

**修改方式：** 对应的自动配置类注册方式从 `spring.factories` 改为 `AutoConfiguration.imports` 文件。

**原方式（为了兼容性，两个可以同时保留）：**

```
org.springframework.boot.autoconfigure.EnableAutoConfiguration=\
com.cloudwise.namingconf.service.CloudwiseExtentManager,\
com.cloudwise.namingconf.service.impl.CloudwiseNamingConfServiceImpl,\
com.cloudwise.namingconf.service.impl.NacosServiceImpl,\
com.cloudwise.namingconf.service.impl.BeanNacos,\
com.cloudwise.namingconf.config.RegisterTextEncryptorListener
```

**迁移到 `META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports`：**

内容如下，注意一个类名一行，不要加换行符：

```
com.cloudwise.namingconf.service.CloudwiseExtentManager
com.cloudwise.namingconf.service.impl.CloudwiseNamingConfServiceImpl
com.cloudwise.namingconf.service.impl.NacosServiceImpl
com.cloudwise.namingconf.service.impl.BeanNacos
com.cloudwise.namingconf.config.RegisterTextEncryptorListener
```

**常见问题：**

有些框架也会出现适配不彻底的问题，例如 `DynamicDataSourceAutoConfiguration` 动态数据源。

**问题现象：** `@com.baomidou.dynamic.datasource.annotation.DS` 注解不生效，动态数据源切换没有效果。

**原因：** `EnableAutoConfiguration` 在 Spring 6 中不再支持。

**解决方案：** 主动加载 `DynamicDataSourceAutoConfiguration`。

在 `META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports` 中增加一行，内容为：

```
com.baomidou.dynamic.datasource.spring.boot.autoconfigure.DynamicDataSourceAutoConfiguration
```

**注意：** `rocketmq-spring-boot` 也存在类似问题，版本升级到 2.3.x 即可（inf-bom 3.x 已升级）。

##### 2.4.4.4 WebSecurityConfigurerAdapter 被移除

**重要：** `org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter` 被正式移除。

**参考文档：** https://spring.io/blog/2022/02/21/spring-security-without-the-websecurityconfigureradapter

**解决方案：** 按照 Spring 提供的说明修改即可，也可直接参考 DOUC 修改后的代码：

https://git.cloudwise.com/DOCP/digital_operation_user_center/-/blob/develop/douc-service/src/main/java/com/cloudwise/douc/service/configuration/ActuatorWebSecurityConfigurationAdapter.java

```java


/**
 * @author bernie.wang
 * @description:
 * @date Created in 10:42 AM 2021/10/14.
 */
@EnableWebSecurity
@Configuration
@Slf4j
public class ActuatorWebSecurityConfigurationAdapter {

    // Admin 链：仅 /admin/**，优先级更高
    @Bean
    @Order(1)
    public SecurityFilterChain adminFilterChain(HttpSecurity http,
                                                HandlerMappingIntrospector mvcHandlerMappingIntrospector,
                                                ObjectMapper objectMapper) throws Exception {
        MvcRequestMatcher adminAll = new MvcRequestMatcher(mvcHandlerMappingIntrospector, "/admin/**");
        MvcRequestMatcher adminApi = new MvcRequestMatcher(mvcHandlerMappingIntrospector, "/admin/api/**");

        http.securityMatcher(adminAll)
                .csrf(AbstractHttpConfigurer::disable)
                .userDetailsService(adminUserDetailsService())
                .httpBasic(AbstractHttpConfigurer::disable)
                .authorizeHttpRequests(auth -> auth
                        .requestMatchers(adminApi).hasRole("ADMIN")
                        .anyRequest().permitAll()
                )
                .exceptionHandling(ex -> ex.authenticationEntryPoint(new CustomAuthenticationEntryPoint()))
                .sessionManagement(s -> s.sessionCreationPolicy(SessionCreationPolicy.IF_REQUIRED))
                .formLogin(form -> form
                        .loginPage("/doucAdminWeb/index.html#/login")
                        .loginProcessingUrl("/admin/auth")
                        .usernameParameter("username")
                        .passwordParameter("password")
                        .successHandler((request, response, authentication) -> {
                            response.setContentType(MediaType.JSON_UTF_8.toString());
                            response.setStatus(HttpServletResponse.SC_OK);
                            Map<String, Object> result = new HashMap<>();
                            result.put("code", 200);
                            result.put("message", "登录成功");
                            result.put("url", "/doucAdminWeb/index.html");
                            response.getWriter().write(objectMapper.writeValueAsString(result));
                        })
                        .failureHandler((request, response, exception) -> {
                            response.setContentType(MediaType.JSON_UTF_8.toString());
                            response.setStatus(HttpServletResponse.SC_OK);
                            Map<String, Object> result = new HashMap<>();
                            result.put("code", 401);
                            result.put("message", "用户名或密码错误");
                            result.put("url", "/doucAdminWeb/index.html#/login?error=true");
                            response.getWriter().write(objectMapper.writeValueAsString(result));
                        })
                )
                .logout(logout -> logout
                        .logoutUrl("/admin/logout")
                        .logoutSuccessHandler((request, response, authentication) -> {
                            response.setContentType(MediaType.JSON_UTF_8.toString());
                            response.setStatus(HttpServletResponse.SC_OK);
                            Map<String, Object> result = new HashMap<>();
                            result.put("code", 200);
                            result.put("message", "登出成功");
                            result.put("url", "/doucAdminWeb/index.html#/login");
                            response.getWriter().write(objectMapper.writeValueAsString(result));
                        })
                        .deleteCookies("SESSION", "JSESSIONID")
                );
        return http.build();
    }

    @Bean
    @Order(2)
    public SecurityFilterChain userFilterChain(HttpSecurity http) throws Exception {
        http
                .securityContext(c -> c.requireExplicitSave(false))
                .authorizeHttpRequests(auth -> auth
                        .requestMatchers("/").authenticated()
                        .requestMatchers("/autoconfig/**",
                                "/beans/**",
                                "/env/**",
                                "/configprops/**",
                                "/dump/**",
                                "/health/**",
                                "/info/**",
                                "/mappings/**",
                                "/metrics/**",
                                "/shutdown/**",
                                "/trace/**",
                                "/v2/api-docs",
                                "/auditevents/**",
                                "/flyway/**",
                                "/heapdump/**",
                                "/httptrace/**",
                                "/jolokia/**",
                                "/logfile/**",
                                "/loggers/**",
                                "/liquibase/**",
                                "/prometheus/**",
                                "/sessions/**",
                                "/threaddump/**",
                                "/swagger-resources/**",
                                "/scheduledtasks/**",
                                "/service-registry/**"
                        ).hasRole("USER")
                        .anyRequest().permitAll()
                )
                .sessionManagement(s -> s.sessionCreationPolicy(SessionCreationPolicy.IF_REQUIRED))
                .csrf(AbstractHttpConfigurer::disable)
                .userDetailsService(userUserDetailsService())
                .httpBasic(Customizer.withDefaults());
        return http.build();
    }

    @Bean
    public UserDetailsService adminUserDetailsService() {
        String userName = ConfigUtils.getString("backendLoginName");
        String passCode = ConfigUtils.getString("backendLoginPass");
        passCode = SecretManager.decryptUnknownKey(passCode);
        log.info("admin userName:{},passCode:{}", userName, passCode);
        UserDetails admin = User.withDefaultPasswordEncoder().username(userName).password(passCode).roles("ADMIN").build();
        //用户名区分大小写
        return new InMemoryUserDetailsManager(admin);
    }


    // 其余 Bean 与原来一致
    @Bean
    public FilterRegistrationBean<ForwardedHeaderFilter> forwardedHeaderFilter() {
        FilterRegistrationBean<ForwardedHeaderFilter> bean = new FilterRegistrationBean<>();
        bean.setFilter(new ForwardedHeaderFilter());
        bean.setOrder(Ordered.HIGHEST_PRECEDENCE + 2);
        return bean;
    }

    @Bean
    public CookieSerializer cookieSerializer() {
        return new PathBasedCookieSerializer();
    }

    @Bean
    public UserDetailsService userUserDetailsService() {
        String userName = ConfigUtils.getString("spring.security.user.name");
        if (userName == null) {
            userName = ConfigUtils.getString("springActuatorUsername");
        }
        String passCode = ConfigUtils.getString("spring.security.user.password");
        if (passCode == null) {
            passCode = ConfigUtils.getString("springActuatorPassword");
        }
        passCode = SecretManager.decryptUnknownKey(passCode);
        UserDetails user = User.withDefaultPasswordEncoder().username(userName).password(passCode).roles("USER").build();
        return new InMemoryUserDetailsManager(user);
    }

}
```

##### 2.4.4.5 @Bean 的返回值不允许使用 void

**问题现象：**

报错：`@Bean method 'xxx' must not be declared as void; change the method's return type or its annotation`

**原因：** `@Bean` 注解的方法返回值不能为 void，如果是初始化业务，应更换为 `@PostConstruct`。

**原代码（Spring 6 不再支持，使用将直接报错）：**

```java

@Bean
public void initElementType() {
    Set<String> accountIds = AccountCache.getTopAccountAndUserMap().keySet();
    accountIds.parallelStream().forEach(accountId -> iComponentInitService.initComponentType(accountId));
}
```

**修改为：**

```java

@PostConstruct
public void initElementType() {
    Set<String> accountIds = AccountCache.getTopAccountAndUserMap().keySet();
    accountIds.parallelStream().forEach(accountId -> iComponentInitService.initComponentType(accountId));
}
```

##### 2.4.4.6 Actuator 模块变更

**原配置：**

```properties
### 禁用所有端口
management.endpoints.enabled-by-default=false
### 开放metrics端口
management.endpoint.prometheus.enabled=true
### 开放health端口
management.endpoint.health.enabled=true
```

变更为

```properties
### 禁用所有端口
management.endpoints.access.default=NONE
### 开放metrics端口
management.endpoint.prometheus.access=READ_ONLY
### 开放health端口
management.endpoint.health.access=READ_ONLY
```

##### 2.4.4.7 HttpClient 版本变更

**重要：** Spring 默认 HttpClient 升级到 HttpClient 5。

**依赖变更：**

```xml

<dependency>
    <groupId>org.apache.httpcomponents.client5</groupId>
    <artifactId>httpclient5</artifactId>
</dependency>
```

**导入包变更：**

| 原导入                                                                    | 新导入                                                                             |
|------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `import org.apache.http.client.config.RequestConfig;`                  | `import org.apache.hc.client5.http.config.RequestConfig;`                       |
| `import org.apache.http.impl.client.CloseableHttpClient;`              | `import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;`           |
| `import org.apache.http.impl.client.HttpClients;`                      | `import org.apache.hc.client5.http.impl.classic.HttpClients;`                   |
| `import org.apache.http.conn.HttpClientConnectionManager;`             | `import org.apache.hc.client5.http.io.HttpClientConnectionManager;`             |
| `import org.apache.http.impl.client.HttpClientBuilder;`                | `import org.apache.hc.client5.http.impl.classic.HttpClientBuilder;`             |
| `import org.apache.http.impl.conn.PoolingHttpClientConnectionManager;` | `import org.apache.hc.client5.http.impl.io.PoolingHttpClientConnectionManager;` |

**注意：** 使用 IDE 自动导包功能即可完成迁移。

##### 2.4.4.8 HandlerInterceptorAdapter 被移除

**原因：** `org.springframework.web.servlet.handler.HandlerInterceptorAdapter` 适配器被移除。

**解决方案：** 实现 `org.springframework.web.servlet.HandlerInterceptor` 接口即可，均为 default 方法，和原有用法一致。

##### 2.4.4.9 beetl-antlr4.13-support 依赖

**问题现象：** 程序运行报错：不支持的 antlr 版本: 4.13.2。

**解决方案：** 联系 xiandafu@126.com 定制，或者参考源码 antlr4.5-support。

**新增依赖：**

```xml

<dependency>
    <groupId>com.ibeetl</groupId>
    <artifactId>beetl-antlr4.13-support</artifactId>
</dependency>
```

#### 2.4.5 常见的 API 变更

**说明：** 各产品业务不同，不一定都会遇到，按需修改即可。

##### 2.4.5.1 BatchErrorHandler 被移除

**原因：** `org.springframework.kafka.listener.BatchErrorHandler` 被移除。

**解决方案：** 修改为实现 `org.springframework.kafka.listener.CommonErrorHandler` 接口，实现 `handleBatch` 方法。

##### 2.4.5.2 setBatchErrorHandler 被移除

**原因：** `org.springframework.kafka.config.ConcurrentKafkaListenerContainerFactory` 中 `setBatchErrorHandler` 方法被移除。

**解决方案：** 修改为 `setCommonErrorHandler`。

##### 2.4.5.3 WebFluxTagsProvider 被移除

**原因：** `org.springframework.boot.actuate.metrics.web.reactive.server.WebFluxTagsProvider` 被移除，需要更换实现。

**原代码示例：**

```java

@Configuration
public class HttpServerWebFluxTagsProvider {

    @Bean
    public WebFluxTagsProvider webFluxTagsProvider() {
        return new WebFluxTagsProvider() {
            @Override
            public Iterable<Tag> httpRequestTags(ServerWebExchange exchange, Throwable exception) {
                //修改默认统计的uri=描述为真实地址
                Tag urlTag = Tag.of("uri", exchange.getRequest().getURI().getPath());
                return Arrays.asList(WebFluxTags.method(exchange), urlTag, WebFluxTags.exception(exception),
                        WebFluxTags.status(exchange), WebFluxTags.outcome(exchange, exception));
            }
        };
    }
}
```

**修改为：**

```java

@Configuration
public class HttpServerWebFluxTagsProvider {
    @Bean
    public ServerRequestObservationConvention customServerRequestObservationConvention() {
        return new DefaultServerRequestObservationConvention() {
            @Override
            public KeyValues getLowCardinalityKeyValues(ServerRequestObservationContext ctx) {
                // method
                String method = ctx.getCarrier().getMethod().name();
                // uri: 真实路径
                String uri = ctx.getCarrier().getURI().getPath();
                // exception
                String exception = (ctx.getError() != null) ? ctx.getError().getClass().getSimpleName() : "None";
                String status = "UNKNOWN";
                Outcome outcome = Outcome.UNKNOWN;
                if (ctx.getResponse() != null && ctx.getResponse().getStatusCode() != null) {
                    HttpStatusCode sc = ctx.getResponse().getStatusCode();
                    status = Integer.toString(sc.value());
                    outcome = Outcome.forStatus(sc.value());
                }
                // 与旧版等价的标签集合与顺序：method, uri, exception, status, outcome
                return KeyValues.of(
                        KeyValue.of("method", method),
                        KeyValue.of("uri", uri),
                        KeyValue.of("exception", exception),
                        KeyValue.of("status", status),
                        KeyValue.of("outcome", outcome.name())
                );
            }
        };
    }
}
```

##### 2.4.5.4 SchedulingConfigurer 方法变更

**原因：** 方法的入参有微调，按提示修改即可。

**原代码示例：**

```java

@Override
public void configureTasks(ScheduledTaskRegistrar taskRegistrar) {
    taskRegistrar.addTriggerTask(this::init, triggerContext -> {
        PeriodicTrigger periodicTrigger = new PeriodicTrigger(Duration.ofSeconds(5L).toMillis());
        return periodicTrigger.nextExecutionTime(triggerContext);
    });
}
```

**修改为：**

```java

@Override
public void configureTasks(ScheduledTaskRegistrar taskRegistrar) {
    taskRegistrar.addTriggerTask(this::init, triggerContext -> {
        PeriodicTrigger periodicTrigger = new PeriodicTrigger(Duration.ofSeconds(5L));
        return periodicTrigger.nextExecution(triggerContext);
    });
}
```

##### 2.4.5.5 getMethodValue 被移除

**原因：** `org.springframework.http.HttpRequest#getMethodValue` 被删除。

**原代码：**

```java
String methodValue = request.getMethodValue();
```

**修改为：**

```java
String methodValue = request.getMethod().name();
```

##### 2.4.5.6 subscriberContext 被移除

**原因：** `reactor.core.publisher.Mono#subscriberContext` 被移除。

**解决方案：** 修改为 `contextWrite` 方法。

##### 2.4.5.7 FieldStrategy.IGNORED 被移除

**原因：** `com.baomidou.mybatisplus.annotation.FieldStrategy.IGNORED` 被移除。

**解决方案：** 修改为 `FieldStrategy.ALWAYS`。

##### 2.4.5.8 AntPathRequestMatcher 被移除

**原代码示例：**

```java
new AntPathRequestMatcher(this.addPrefixRedirectUrl("/douc/api/v1/sso/saml/metadata/**")
```

**修改为：**

```java
chains.add(new DefaultSecurityFilterChain(PathPatternRequestMatcher.withDefaults().

matcher(this.addPrefixRedirectUrl("/douc/api/v1/sso/saml/metadata/**")),

metadataDisplayFilter()));
```

##### 2.4.5.9 KafkaTemplate 返回值变更

**原因：** `KafkaTemplate` 返回值类型变更。

**原类型：** `ListenableFuture<SendResult<K, V>>`

**新类型：** `CompletableFuture<SendResult<K, V>>`

##### 2.4.5.10 找不到 PaginationInnerInterceptor

**问题现象：** `com.baomidou.mybatisplus.extension.plugins.inner.PaginationInnerInterceptor` 找不到。

**原因：** MyBatis-Plus 拆分了新依赖，已迁移到新坐标，全限定名不变。

**解决方案：** 新增以下依赖：

```xml

<dependency>
    <groupId>com.baomidou</groupId>
    <artifactId>mybatis-plus-jsqlparser</artifactId>
</dependency>
```

##### 2.4.5.11 PropertyNamingStrategy.SNAKE_CASE 被移除

**原因：** `com.fasterxml.jackson.databind.PropertyNamingStrategy.SNAKE_CASE` 已移除。

**解决方案：** 替换为 `com.fasterxml.jackson.databind.PropertyNamingStrategies.SNAKE_CASE`。

##### 2.4.5.12 HttpMethod.resolve 被移除

**原因：** `org.springframework.http.HttpMethod.resolve` 已删除。

**解决方案：** 替换为 `HttpMethod.valueOf`。

**注意：** 请确保保证参数的合法性。

```java
switch(method) {
    case "GET" -> var10000 = GET;
    case "HEAD" -> var10000 = HEAD;
    case "POST" -> var10000 = POST;
    case "PUT" -> var10000 = PUT;
    case "PATCH" -> var10000 = PATCH;
    case "DELETE" -> var10000 = DELETE;
    case "OPTIONS" -> var10000 = OPTIONS;
    case "TRACE" -> var10000 = TRACE;
    default -> var10000 = new HttpMethod(method);
}
```

##### 2.4.5.13 Base64Utils 被移除

**原因：** `org.springframework.util.Base64Utils` 被移除。

**解决方案：** 用 JDK 自带的 `java.util.Base64` 替代，功能等价且更标准。

**常用映射：**

- `Base64Utils.encodeToString(bytes)` → `Base64.getEncoder().encodeToString(bytes)`
- `Base64Utils.decodeFromString(str)` → `Base64.getDecoder().decode(str)`
- `Base64Utils.encode(bytes)` → `Base64.getEncoder().encode(bytes)`
- `Base64Utils.decode(bytes)` → `Base64.getDecoder().decode(bytes)`
- `Base64Utils.encodeToUrlSafeString(bytes)` → `Base64.getUrlEncoder().withoutPadding().encodeToString(bytes)`
- URL Safe 解码 → `Base64.getUrlDecoder().decode(str)`

##### 2.4.5.14 PrometheusMeterRegistry 被移除

**原因：** `io.micrometer.prometheus.PrometheusMeterRegistry` 被移除。

**解决方案：** 迁移到 `io.micrometer.prometheusmetrics.PrometheusMeterRegistry`。

##### 2.4.5.15 DataSourceProperty 被移除

**原因：** `com.baomidou.dynamic.datasource.spring.boot.autoconfigure.DataSourceProperty` 被移除。

**解决方案：** 迁移到 `com.baomidou.dynamic.datasource.creator.DataSourceProperty`。

##### 2.4.5.16 DruidConfig 被移除

**原因：** `com.baomidou.dynamic.datasource.spring.boot.autoconfigure.druid.DruidConfig` 被移除。

**解决方案：** 迁移到 `com.baomidou.dynamic.datasource.creator.druid.DruidConfig`。

##### 2.4.5.17 RocketMQMessageConvert 实现类必须有无参构造方法

**问题现象：**

```
Description:
Parameter 0 of constructor in com.cloudwise.message.service.rocketmq.configuration.MyListenerContainerConfiguration required a bean of type 'org.apache.rocketmq.spring.support.RocketMQMessageConverter' that could not be found.
Action:
Consider defining a bean of type 'org.apache.rocketmq.spring.support.RocketMQMessageConverter' in your configuration.
```

**解决方案：** 实现类增加无参构造方法即可，无需其他修改。

##### 2.4.5.18 IRepository#update 方法被移除

**原因：** 过期方法已删除。

**原代码：**

```java

@Deprecated
default boolean saveOrUpdate(T entity, Wrapper<T> updateWrapper) {
    return update(entity, updateWrapper) || saveOrUpdate(entity);
}
```

**解决方案：** 修改为 `update(entity, updateWrapper) || saveOrUpdate(entity);` 即可。

##### 2.4.5.19 cannot be cast to class 类型转换异常

**问题现象：**

报错：
`java.lang.ClassCastException: class com.cloudwise.douc.service.model.department.DepartmentNodeCacheDTO cannot be cast to class com.cloudwise.douc.metadata.model.department.DepartmentNode (com.cloudwise.douc.service.model.department.DepartmentNodeCacheDTO and com.cloudwise.douc.metadata.model.department.DepartmentNode are in unnamed module of loader 'app')`

**原因：** 实际类型和声明类型不一致，在
`departmentNode = BeanDeepCopyUtil.copyProperties(departNodes.get(key), DepartmentNode.class);` 泛型擦除后，`HashMap#get`
的实际返回是 Object 会抛出类型异常。

JDK8 编译的版本显示正常的原因为，JDK8 类型断言更靠后，JDK17 断言更靠前。

**代码编译差异：**

**JDK8：**

```java
departmentNode =(DepartmentNode)BeanDeepCopyUtil.

copyProperties(departNodes.get(key),DepartmentNode.class);
```

JDK17

```java
departmentNode =(DepartmentNode)BeanDeepCopyUtil.

copyProperties((DepartmentNode)departNodes.

get(key),DepartmentNode.class);
```

**解决方案：** 泛型声明和实际类型保持一致。

**注意：** 该问题和 Spring 升级无关，是 JDK17 和 JDK8 断言机制变更导致的，JDK17 断言更严格更靠前。

##### 2.4.5.21 Checker Framework 注解依赖缺失

**重要：** 升级到 Spring 6/Hibernate 7 后，使用 Caffeine 缓存库时，Checker Framework 注解不再自动包含。

**问题现象：**

编译错误：

- `程序包org.checkerframework.checker.index.qual不存在`
- `程序包org.checkerframework.checker.nullness.qual不存在`

**原因：**

Caffeine 缓存库的接口方法使用了 Checker Framework 注解（如 `@NonNull`、`@NonNegative`、`@Nullable`），这些注解在升级后需要显式添加依赖。

**解决方案：**

在使用 Caffeine 的模块的 `pom.xml` 中添加 `checker-qual` 依赖：

```xml

<dependency>
    <groupId>com.github.ben-manes.caffeine</groupId>
    <artifactId>caffeine</artifactId>
</dependency>
<!-- Checker Framework annotations for Caffeine -->
<dependency>
    <groupId>org.checkerframework</groupId>
    <artifactId>checker-qual</artifactId>
</dependency>
```

**常见使用场景：**

1. **Caffeine 缓存配置类**

```java
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.nullness.qual.NonNull;

public class CustomCacheConfig {
    @Bean
    public Cache<String, Object> caffeineCache() {
        return Caffeine.newBuilder()
                .expireAfter(new Expiry<String, Object>() {
                    @Override
                    public long expireAfterCreate(
                            @NonNull String key,
                            @NonNull Object value, long currentTime) {
                        return TimeUnit.SECONDS.toNanos(60);
                    }

                    @Override
                    public long expireAfterUpdate(
                            @NonNull String key,
                            @NonNull Object value, long currentTime,
                            @NonNegative long currentDuration) {
                        return currentDuration;
                    }
                }).build();
    }
}
```

2. **泛型类型注解**

```java
import org.checkerframework.checker.nullness.qual.Nullable;

Map<@Nullable String, @Nullable Object> map = Maps.newHashMap();
```

**注意事项：**

- `checker-qual` 依赖版本由 `inf-bom` 统一管理，无需指定版本号
- 这些注解主要用于静态代码分析，不影响运行时行为
- 如果项目不使用 Caffeine 或相关注解，可以忽略此问题

##### 2.4.5.22 XXL-Job ReturnT.SUCCESS 变更

**重要：** 升级到 XXL-Job 3.2.0 后，`ReturnT.SUCCESS` 静态常量已被移除。

**问题现象：**

编译错误：

- `找不到符号 符号: 变量 SUCCESS 位置: 类 com.xxl.job.core.biz.model.ReturnT`

**原因：**

XXL-Job 3.2.0 中，`ReturnT.SUCCESS` 静态常量已被移除，改为使用 `ReturnT.ofSuccess()` 方法。

**解决方案：**

将所有 `ReturnT.SUCCESS` 替换为 `ReturnT.ofSuccess()`。

**原代码示例：**

```java

@XxlJob("demoJobHandler")
public ReturnT<String> demoJobHandler() {
    // 业务逻辑
    return ReturnT.SUCCESS;
}
```

**修改为：**

```java

@XxlJob("demoJobHandler")
public ReturnT<String> demoJobHandler() {
    // 业务逻辑
    return ReturnT.ofSuccess();
}
```

**关键变更点：**

1. `ReturnT.SUCCESS` → `ReturnT.ofSuccess()`
2. 功能完全等价，`ofSuccess()` 方法返回 code 为 `SUCCESS_CODE` 的 `ReturnT` 实例

**注意事项：**

- `ReturnT.ofSuccess()` 每次返回新的实例，而原来的 `SUCCESS` 是单例常量
- 如果需要返回带消息的成功结果，可以使用 `ReturnT.ofSuccess("消息内容")`
- 如果任务执行失败，使用 `ReturnT.ofFail("错误信息")` 或 `new ReturnT<>(ReturnT.FAIL_CODE, "错误信息")`
- **重要：** `ReturnT.SUCCESS_CODE` 是常量，**不能**被替换。批量替换时需要注意区分：
    - `ReturnT.SUCCESS` → `ReturnT.ofSuccess()`（需要替换）
    - `ReturnT.SUCCESS_CODE` → 保持不变（常量，不能替换）

**错误示例：**

```java
// ❌ 错误：将常量也替换了
if(null != res && ReturnT.ofSuccess()_CODE == res.getCode()) {
    // ...
}
```

**正确示例：**

```java
// ✅ 正确：只替换 SUCCESS，保留 SUCCESS_CODE 常量
if(null != res && ReturnT.SUCCESS_CODE == res.getCode()) {
    // ...
}
```

##### 2.4.5.23 Lombok var 导入冲突

**重要：** 升级到 Java 17 后，`import lombok.var;` 会导致编译错误。

**问题现象：**

编译错误：

- `非法引用受限类型 'var'，import lombok.var;`

**原因：**

在 Java 10+ 中，`var` 是保留关键字，用于局部变量类型推断。Lombok 的 `var` 功能在 Java 10+ 中已经不再需要，因为 Java 本身支持了
`var` 关键字。在 Java 17 中，如果导入 `lombok.var`，会与 Java 的 `var` 关键字冲突，导致编译错误。

**解决方案：**

删除所有 `import lombok.var;` 语句。代码中的 `var` 会自动使用 Java 原生的 `var` 关键字，功能完全一致。

**原代码示例：**

```java
import lombok.var;

public class Example {
    public void method() {
        var list = new ArrayList<String>();
        for (var item : list) {
            // ...
        }
    }
}
```

**修改为：**

```java
// 删除 import lombok.var; 即可

public class Example {
    public void method() {
        var list = new ArrayList<String>();  // 使用 Java 原生的 var
        for (var item : list) {
            // ...
        }
    }
}
```

**关键变更点：**

1. 删除所有 `import lombok.var;` 语句
2. 代码中的 `var` 会自动使用 Java 10+ 原生的局部变量类型推断
3. 功能完全等价，无需修改其他代码

**注意事项：**

- Java 10+ 原生的 `var` 只能用于局部变量，不能用于方法参数、返回值、字段等
- Lombok 的 `var` 功能在 Java 10+ 中已废弃，建议使用 Java 原生的 `var`
- 如果代码中使用了 `var` 作为变量名（非类型推断），需要重命名变量

##### 2.4.5.24 LsdkV1 方法返回类型变更

**重要：** 升级后，`LsdkV1.lckCurrentNum` 和 `LsdkV1.lckNum` 方法的返回类型从 `int` 变更为 `long`。

**问题现象：**

编译错误：

- `Type mismatch: cannot convert from long to int`
- `Type mismatch: cannot convert from long to int`（`LsdkV1.lckCurrentNum`）
- `Type mismatch: cannot convert from long to int`（`LsdkV1.lckNum`）

**原因：**

升级后，`LsdkV1.lckCurrentNum` 和 `LsdkV1.lckNum` 方法的返回类型从 `int` 变更为 `long`，导致类型不匹配。

**解决方案：**

将所有使用这些方法的变量类型从 `int` 改为 `long`，并更新相关方法的返回类型。

**原代码示例：**

```java
private int getCiLicenseTotalNum() {
    int totalLicenseNum = 0;
    try {
        totalLicenseNum = LsdkV1.lckNum(ciLicenseFeature);
    } catch (LckFeatureNotFoundException e) {
        // ...
    }
    return totalLicenseNum;
}

public long ciLicenseRemainder() {
    try {
        int currentNum = LsdkV1.lckCurrentNum(ciLicenseFeature);
        int totalNum = getCiLicenseTotalNum();
        // ...
        return currentNum;
    } catch (Exception e) {
        // ...
    }
    return 0;
}

public long ciLicenseUsed() {
    int usedLicenseNum = 0;
    try {
        int totalNum = getCiLicenseTotalNum();
        int currentNum = LsdkV1.lckCurrentNum(ciLicenseFeature);
        usedLicenseNum = totalNum - currentNum;
    } catch (Exception e) {
        // ...
    }
    return usedLicenseNum;
}
```

**修改为：**

```java
private long getCiLicenseTotalNum() {
    long totalLicenseNum = 0;
    try {
        totalLicenseNum = LsdkV1.lckNum(ciLicenseFeature);
    } catch (LckFeatureNotFoundException e) {
        // ...
    }
    return totalLicenseNum;
}

public long ciLicenseRemainder() {
    try {
        long currentNum = LsdkV1.lckCurrentNum(ciLicenseFeature);
        long totalNum = getCiLicenseTotalNum();
        // ...
        return currentNum;
    } catch (Exception e) {
        // ...
    }
    return 0;
}

public long ciLicenseUsed() {
    long usedLicenseNum = 0;
    try {
        long totalNum = getCiLicenseTotalNum();
        long currentNum = LsdkV1.lckCurrentNum(ciLicenseFeature);
        usedLicenseNum = totalNum - currentNum;
    } catch (Exception e) {
        // ...
    }
    return usedLicenseNum;
}
```

**关键变更点：**

1. `LsdkV1.lckCurrentNum` 返回类型：`int` → `long`
2. `LsdkV1.lckNum` 返回类型：`int` → `long`
3. 所有接收这些方法返回值的变量类型需要从 `int` 改为 `long`
4. 相关方法的返回类型也需要相应调整

**注意事项：**

- 不要使用类型转换（如 `(int)`），应该直接使用 `long` 类型
- 如果方法返回类型是 `long`，调用该方法的变量也应该使用 `long` 类型
- 检查所有使用这些方法的地方，确保类型一致性

#### 2.4.6 Spring Security & Spring Gateway 相关变更

**说明：** Spring 6 升级后，Security 和 Gateway 相关 API 变化较大，移除了大量过期 API 和实现方式，下面是常见问题，可参考。如未涉及无需修改。

##### 2.4.6.1 CONTENT_TYPE 变更

**问题现象：**

报错：
`No converter for [class com.cloudwise.docp.mobile.model.common.ResultModel] with preset Content-Type 'text/html;charset=utf-8']`

**请求报错，返回如下：**

```json
{
  "timestamp": "2025-08-12T08:55:01.872+00:00",
  "status": 500,
  "error": "Internal Server Error",
  "path": "/api/docp/mobile/service/appManage/getAppList"
}
```

**后台报错：**

```
2025-08-12 16:55:01.842|WARN  |36765|http-nio-18582-exec-1|org.springframework.web.servlet.handler.AbstractHandlerExceptionResolver.logException(AbstractHandlerExceptionResolver.java:254)|Resolved [org.springframework.http.converter.HttpMessageNotWritableException: No converter for [class com.cloudwise.docp.mobile.model.common.ResultModel] with preset Content-Type 'text/html;charset=utf-8']
```

**原因：** `org.springframework.http.server.ServletServerHttpResponse.ServletResponseHttpHeaders` 调整了获取 CONTENT_TYPE
的实现方式。

**Spring 6 实现：**

```java

@Override
@Nullable
public String getFirst(String headerName) {
    if (headerName.equalsIgnoreCase(CONTENT_TYPE)) {
        // Content-Type is written as an override so check super first
        String value = super.getFirst(headerName);
        return (value != null ? value : servletResponse.getContentType());
    } else {
        String value = servletResponse.getHeader(headerName);
        return (value != null ? value : super.getFirst(headerName));
    }
}
```

**Spring 5 实现：**

```java

@Override
@Nullable
public String getFirst(String headerName) {
    if (headerName.equalsIgnoreCase(CONTENT_TYPE)) {
        // Content-Type is written as an override so check super first
        String value = super.getFirst(headerName);
        return (value != null ? value : servletResponse.getHeader(headerName));
    } else {
        String value = servletResponse.getHeader(headerName);
        return (value != null ? value : super.getFirst(headerName));
    }
}
```

**解决方案：** 如果代码中有调用如下代码，则会出现 Spring 5 正常，Spring 6 不可用的情况，删除即可：

```java
response.setContentType("text/html;charset=utf-8");
```

##### 2.4.6.2 ServerHttpResponse 返回值变更

**变更内容：**

1. `org.springframework.http.server.reactive.ServerHttpResponse` 中 `HttpStatus getStatusCode();` 方法变更为
   `HttpStatusCode getStatusCode();`
    - 其中 `HttpStatus` 为 `HttpStatusCode` 的实现，实际未变，修改接收类型即可。

2. `getRawStatusCode` 已弃用，变更为 `in favor of getStatusCode(), for removal in 7.0`

##### 2.4.6.3 subscriberContext 被移除

**原因：** `reactor.core.publisher.Mono` 的 `subscriberContext` 方法被移除。

**解决方案：** 变更为 `contextWrite`。

##### 2.4.6.4 ResponseStatusException 部分方法被移除

**原因：** `org.springframework.web.server.ResponseStatusException` 的 `getStatus` 方法被删除。

**解决方案：** 变更为 `getStatusCode`。

##### 2.4.6.5 spring-cloud-starter-gateway 已过期

**问题现象：**
`spring-cloud-starter-gateway is deprecated. It will be removed in the next major release. Please use spring-cloud-starter-gateway-server-webflux instead.`

**原依赖：**

```xml

<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-gateway</artifactId>
</dependency>
```

**修改为：**

```xml

<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-gateway-server-webflux</artifactId>
</dependency>
```

##### 2.4.6.6 Security 序列化 ID 变更

**问题现象：**

报以下错误：

```
Caused by: java.io.InvalidClassException: org.springframework.security.core.context.SecurityContextImpl; local class incompatible: stream classdesc serialVersionUID = 580, local class serialVersionUID = 620
Caused by: java.io.InvalidClassException: org.springframework.security.core.authority.SimpleGrantedAuthority; local class incompatible: stream classdesc serialVersionUID = 580, local class serialVersionUID = 620
```

**源代码：**

```java
public class SecurityContextImpl implements SecurityContext {
    private static final long serialVersionUID = SpringSecurityCoreVersion.SERIAL_VERSION_UID;
    private Authentication authentication;

    public SecurityContextImpl() {
    }
    .............
}

public final class SimpleGrantedAuthority implements GrantedAuthority {
    private static final long serialVersionUID = SpringSecurityCoreVersion.SERIAL_VERSION_UID;
 .........
}

public final class SpringSecurityCoreVersion {
    private static final String DISABLE_CHECKS = SpringSecurityCoreVersion.class.getName().concat(".DISABLE_CHECKS");
    private static final Log logger = LogFactory.getLog(SpringSecurityCoreVersion.class);
    /**
     * Global Serialization value for Spring Security classes.
     */
    public static final long SERIAL_VERSION_UID = 620L;
    ............
}
```

**原因：** 原来在代码中本地重写了 `SpringSecurityCoreVersion` 类：

```java
public final class SpringSecurityCoreVersion {
    private static final String DISABLE_CHECKS = SpringSecurityCoreVersion.class.getName().concat(".DISABLE_CHECKS");
    private static final Log logger = LogFactory.getLog(SpringSecurityCoreVersion.class);
    /**
     * Global Serialization value for Spring Security classes.
     * <p>
     * N.B. Classes are not intended to be serializable between different versions. See SEC-1709 for why we still need a
     * serial version.
     */
    public static final long SERIAL_VERSION_UID = 530L;
    ..............
}
```

**解决方案：** 但实际不生效，因为 Java 在编译期间做了常量折叠优化，编译时已经将已知的常量表达式替换为其结果值，覆盖
`SpringSecurityCoreVersion` 常量是没有用的。

- **短期方案：** 可覆盖 `SecurityContextImpl`、`SimpleGrantedAuthority` 解决
- **长期方案：** 应该不使用 JDK 序列化保存 Security，改为使用 JSON 等

##### 2.4.6.7 exchange.getRequest().mutate().header 不再有效

**问题现象：** 原有代码，修改 `ServerWebExchange` 对象，如修改 headers：

```java
//将现在的request，添加当前身份
ServerHttpRequest mutableReq = exchange.getRequest().mutate().headers(httpHeaders).build();
exchange.mutate().request(mutableReq).build();
```

**原因：** Spring 5.x 是有效的，Spring 6 不再有效，会生成新的 `ServerWebExchange`，往下传递的依然是原有的
`ServerWebExchange` 对象。

**解决方案：** `ReactiveAuthorizationManager` 只能校验，不能修改请求，将该部分逻辑移动到 `ReactiveAuthorizationManager` 之后的
`WebFilter` 之后执行即可。

**新增拦截器：**

```java

@Slf4j
public class ModifyRequestHeaderFilter implements WebFilter {
    @Override
    public Mono<Void> filter(ServerWebExchange exchange, WebFilterChain chain) {
        return ReactiveSecurityContextHolder.getContext()
                .map(SecurityContext::getAuthentication)
                .map(Authentication::getPrincipal)
                .cast(SessionDetail.class)
                .map(sd -> setHeader(sd, exchange))          // 构造"新的" exchange
                .flatMap(mutated -> chain.filter(mutated))   // 继续传递"新的" exchange
                .switchIfEmpty(Mono.defer(() -> {
                    exchange.getResponse().setStatusCode(HttpStatus.UNAUTHORIZED);
                    return exchange.getResponse().setComplete();
                }))      // 没有 SessionDetail 就原样放行
                .onErrorResume(ClassCastException.class, e -> chain.filter(exchange));
    }

    private ServerWebExchange setHeader(SessionDetail user, ServerWebExchange exchange) {
    ......
        return exchange.mutate().request(mutableReq).build();
    }
}
```

**显示指定到 ReactiveAuthorizationManager 之后执行：**

```java

@Bean
public SecurityWebFilterChain springSecurityFilterChain(final ServerHttpSecurity http) {
    http.cors().and()
            //关闭csrf
            .csrf().disable()
            //关闭表单登录
            .formLogin().disable()
            .................
            .addFilterAfter(modifyRequestHeaderFilter(), SecurityWebFiltersOrder.AUTHORIZATION);
    return http.build();
}
```

**注意：** 其它类似的也一样修改，把 `ServerWebExchange` 传递下去。

##### 2.4.6.8 RedisSessionRepository 未注入

**问题现象：**

报错：
`No qualifying bean of type 'org.springframework.session.data.redis.RedisIndexedSessionRepository' available: expected at least 1 bean which qualifies as autowire candidate.`

**原因：** 在 Spring 5 中，`RedisHttpSessionConfiguration` 会注入 `RedisSessionRepository`。

```java
org.springframework.session.data.redis.config.annotation.web.http.RedisHttpSessionConfiguration

@Bean
@Override
public RedisSessionRepository sessionRepository() {
    RedisTemplate<String, Object> redisTemplate = createRedisTemplate();
    RedisSessionRepository sessionRepository = new RedisSessionRepository(redisTemplate);
    sessionRepository.setDefaultMaxInactiveInterval(getMaxInactiveInterval());
    if (StringUtils.hasText(getRedisNamespace())) {
        sessionRepository.setRedisKeyNamespace(getRedisNamespace());
    }
    sessionRepository.setFlushMode(getFlushMode());
    sessionRepository.setSaveMode(getSaveMode());
    sessionRepository.setSessionIdGenerator(this.sessionIdGenerator);
    getSessionRepositoryCustomizers()
            .forEach((sessionRepositoryCustomizer) -> sessionRepositoryCustomizer.customize(sessionRepository));
    return sessionRepository;
}
```

**解决方案：** Spring 6/Boot 3 后，`spring.session.store-type` 属性配置被移除，且不再默认注册 `RedisSessionRepository`
，显示指定即可。

**方式一：增加注解 `@EnableRedisIndexedHttpSession`**

```java
import org.springframework.session.data.redis.config.annotation.web.http.EnableRedisIndexedHttpSession;

@EnableRedisIndexedHttpSession
```

**方式二：增加配置**

```properties
spring.session.redis.repository-type=INDEXED
```

